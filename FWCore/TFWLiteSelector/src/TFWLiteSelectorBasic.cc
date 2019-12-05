// -*- C++ -*-

//
// Package:     TFWLiteSelector
// Class  :     TFWLiteSelectorBasic
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Jun 27 17:58:10 EDT 2006
//

// user include files
#include "FWCore/TFWLiteSelector/interface/TFWLiteSelectorBasic.h"

#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "DataFormats/Common/interface/setIsMergeable.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/ProductProvenanceRetriever.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventEntryDescription.h"  // kludge to allow compilation
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/FriendlyName.h"
#include "FWCore/Reflection/interface/ObjectWithDict.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

// system include files
#include "TBranch.h"
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace edm {
  namespace root {
    class FWLiteDelayedReader : public DelayedReader {
    public:
      FWLiteDelayedReader() : entry_(-1), eventTree_(nullptr) {}
      void setEntry(Long64_t iEntry) { entry_ = iEntry; }
      void setTree(TTree* iTree) { eventTree_ = iTree; }
      void set(std::shared_ptr<std::unordered_map<unsigned int, BranchDescription const*>> iMap) {
        bidToDesc_ = std::move(iMap);
      }

    private:
      std::unique_ptr<WrapperBase> getTheProduct(BranchID const& k) const;
      std::unique_ptr<WrapperBase> getProduct_(BranchID const& k, EDProductGetter const* ep) override;
      virtual std::unique_ptr<EventEntryDescription> getProvenance_(BranchKey const&) const {
        return std::unique_ptr<EventEntryDescription>();
      }
      void mergeReaders_(DelayedReader*) override {}
      void reset_() override {}

      signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadFromSourceSignal()
          const override {
        return nullptr;
      }
      signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadFromSourceSignal()
          const override {
        return nullptr;
      };

      Long64_t entry_;
      TTree* eventTree_;
      std::shared_ptr<std::unordered_map<unsigned int, BranchDescription const*>> bidToDesc_;
    };

    std::unique_ptr<WrapperBase> FWLiteDelayedReader::getProduct_(BranchID const& k, EDProductGetter const* /*ep*/) {
      return getTheProduct(k);
    }

    std::unique_ptr<WrapperBase> FWLiteDelayedReader::getTheProduct(BranchID const& k) const {
      auto itFind = bidToDesc_->find(k.id());
      if (itFind == bidToDesc_->end()) {
        throw Exception(errors::ProductNotFound) << "could not find entry for product " << k;
      }
      BranchDescription const& bDesc = *(itFind->second);

      TBranch* branch = eventTree_->GetBranch(bDesc.branchName().c_str());
      if (nullptr == branch) {
        throw cms::Exception("MissingBranch") << "could not find branch named '" << bDesc.branchName() << "'"
                                              << "\n Perhaps the data being requested was not saved in this file?";
      }
      //find the class type
      std::string const fullName = wrappedClassName(bDesc.className());
      TypeWithDict classType = TypeWithDict::byName(fullName);
      if (!bool(classType)) {
        throw cms::Exception("MissingDictionary") << "could not find dictionary for type '" << fullName << "'"
                                                  << "\n Please make sure all the necessary libraries are available.";
      }

      //create an instance of it
      ObjectWithDict wrapperObj = classType.construct();
      if (nullptr == wrapperObj.address()) {
        throw cms::Exception("FailedToCreate") << "could not create an instance of '" << fullName << "'";
      }
      void* address = wrapperObj.address();
      branch->SetAddress(&address);
      ObjectWithDict edProdObj = wrapperObj.castObject(TypeWithDict::byName("edm::WrapperBase"));

      WrapperBase* prod = reinterpret_cast<WrapperBase*>(edProdObj.address());

      if (nullptr == prod) {
        throw cms::Exception("FailedConversion") << "failed to convert a '" << fullName << "' to a edm::WrapperBase."
                                                 << "Please contact developers since something is very wrong.";
      }
      branch->GetEntry(entry_);
      return std::unique_ptr<WrapperBase>(prod);
    }

    struct TFWLiteSelectorMembers {
      TFWLiteSelectorMembers()
          : tree_(nullptr),
            reg_(new ProductRegistry()),
            bidToDesc_(std::make_shared<std::unordered_map<unsigned int, BranchDescription const*>>()),
            phreg_(new ProcessHistoryRegistry()),
            branchIDListHelper_(new BranchIDListHelper()),
            // Note that thinned collections are not supported yet, the next
            // line just makes it compile but when the Ref or Ptr tries to
            // find the thinned collection it will report them not found.
            // More work needed here if this is needed (we think no one
            // is using TFWLiteSelector anymore and intend to implement
            // this properly if it turns out we are wrong)
            thinnedAssociationsHelper_(new ThinnedAssociationsHelper()),
            processNames_(),
            reader_(new FWLiteDelayedReader),
            prov_(),
            pointerToBranchBuffer_(),
            provRetriever_(new edm::ProductProvenanceRetriever(0)) {
        reader_->set(get_underlying_safe(bidToDesc_));
      }
      void setTree(TTree* iTree) {
        tree_ = iTree;
        reader_->setTree(iTree);
      }

      TTree const* tree() const { return get_underlying_safe(tree_); }
      TTree*& tree() { return get_underlying_safe(tree_); }
      std::shared_ptr<ProductRegistry const> reg() const { return get_underlying_safe(reg_); }
      std::shared_ptr<ProductRegistry>& reg() { return get_underlying_safe(reg_); }
      std::shared_ptr<BranchIDListHelper const> branchIDListHelper() const {
        return get_underlying_safe(branchIDListHelper_);
      }
      std::shared_ptr<BranchIDListHelper>& branchIDListHelper() { return get_underlying_safe(branchIDListHelper_); }
      std::shared_ptr<ThinnedAssociationsHelper const> thinnedAssociationsHelper() const {
        return get_underlying_safe(thinnedAssociationsHelper_);
      }
      std::shared_ptr<ThinnedAssociationsHelper>& thinnedAssociationsHelper() {
        return get_underlying_safe(thinnedAssociationsHelper_);
      }

      edm::propagate_const<TTree*> tree_;
      edm::propagate_const<std::shared_ptr<ProductRegistry>> reg_;
      edm::propagate_const<std::shared_ptr<std::unordered_map<unsigned int, BranchDescription const*>>> bidToDesc_;
      edm::propagate_const<std::shared_ptr<ProcessHistoryRegistry>> phreg_;
      edm::propagate_const<std::shared_ptr<BranchIDListHelper>> branchIDListHelper_;
      edm::propagate_const<std::shared_ptr<ThinnedAssociationsHelper>> thinnedAssociationsHelper_;
      ProcessHistory processNames_;
      edm::propagate_const<std::shared_ptr<FWLiteDelayedReader>> reader_;
      std::vector<EventEntryDescription> prov_;
      std::vector<EventEntryDescription const*> pointerToBranchBuffer_;
      FileFormatVersion fileFormatVersion_;

      edm::propagate_const<std::shared_ptr<edm::ProductProvenanceRetriever>> provRetriever_;
      edm::ProcessConfiguration pc_;
      edm::propagate_const<std::shared_ptr<edm::EventPrincipal>> ep_;
      edm::ModuleDescription md_;
    };
  }  // namespace root
}  // namespace edm

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TFWLiteSelectorBasic::TFWLiteSelectorBasic() : m_(new edm::root::TFWLiteSelectorMembers), everythingOK_(false) {}

// TFWLiteSelectorBasic::TFWLiteSelectorBasic(TFWLiteSelectorBasic const& rhs)
// {
//    // do actual copying here;
// }

TFWLiteSelectorBasic::~TFWLiteSelectorBasic() {}

//
// assignment operators
//
// TFWLiteSelectorBasic const& TFWLiteSelectorBasic::operator=(TFWLiteSelectorBasic const& rhs)
// {
//   //An exception safe implementation is
//   TFWLiteSelectorBasic temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void TFWLiteSelectorBasic::Begin(TTree* iTree) {
  Init(iTree);
  begin(fInput);
}

void TFWLiteSelectorBasic::SlaveBegin(TTree* iTree) {
  Init(iTree);
  preProcessing(fInput, *fOutput);
}

void TFWLiteSelectorBasic::Init(TTree* iTree) {
  if (iTree == nullptr)
    return;
  m_->setTree(iTree);
}

Bool_t TFWLiteSelectorBasic::Notify() {
  //std::cout << "Notify start" << std::endl;
  //we have switched to a new file
  //get new file from Tree
  if (nullptr == m_->tree_) {
    std::cout << "No tree" << std::endl;
    return kFALSE;
  }
  TFile* file = m_->tree_->GetCurrentFile();
  if (nullptr == file) {
    //When in Rome, do as the Romans
    TChain* chain = dynamic_cast<TChain*>(m_->tree());
    if (nullptr == chain) {
      std::cout << "No file" << std::endl;
      return kFALSE;
    }
    file = chain->GetFile();
    if (nullptr == file) {
      std::cout << "No file" << std::endl;
      return kFALSE;
    }
  }
  setupNewFile(*file);
  return everythingOK_ ? kTRUE : kFALSE;
}

namespace {
  struct Operate {
    Operate(edm::EDProductGetter const* iGetter) : old_(setRefCoreStreamer(iGetter)) {}

    ~Operate() { setRefCoreStreamer(old_); }

  private:
    edm::EDProductGetter const* old_;
  };
}  // namespace

Bool_t TFWLiteSelectorBasic::Process(Long64_t iEntry) {
  //std::cout << "Process start" << std::endl;
  if (everythingOK_) {
    std::unique_ptr<edm::EventAuxiliary> eaux = std::make_unique<edm::EventAuxiliary>();
    edm::EventAuxiliary& aux = *eaux;
    edm::EventAuxiliary* pAux = eaux.get();
    TBranch* branch = m_->tree_->GetBranch(edm::BranchTypeToAuxiliaryBranchName(edm::InEvent).c_str());

    branch->SetAddress(&pAux);
    branch->GetEntry(iEntry);

    //NEW      m_->processNames_ = aux.processHistory();

    //      std::cout << "ProcessNames\n";
    //      for(auto const& name : m_->processNames_) {
    //         std::cout << "  " << name << std::endl;
    //     }

    edm::EventSelectionIDVector eventSelectionIDs;
    edm::EventSelectionIDVector* pEventSelectionIDVector = &eventSelectionIDs;
    TBranch* eventSelectionsBranch = m_->tree_->GetBranch(edm::poolNames::eventSelectionsBranchName().c_str());
    if (!eventSelectionsBranch) {
      throw edm::Exception(edm::errors::FatalRootError) << "Failed to find event Selections branch in event tree";
    }
    eventSelectionsBranch->SetAddress(&pEventSelectionIDVector);
    eventSelectionsBranch->GetEntry(iEntry);

    edm::BranchListIndexes branchListIndexes;
    edm::BranchListIndexes* pBranchListIndexes = &branchListIndexes;
    TBranch* branchListIndexBranch = m_->tree_->GetBranch(edm::poolNames::branchListIndexesBranchName().c_str());
    if (!branchListIndexBranch) {
      throw edm::Exception(edm::errors::FatalRootError) << "Failed to find branch list index branch in event tree";
    }
    branchListIndexBranch->SetAddress(&pBranchListIndexes);
    branchListIndexBranch->GetEntry(iEntry);
    m_->branchIDListHelper_->fixBranchListIndexes(branchListIndexes);

    try {
      m_->reader_->setEntry(iEntry);
      auto runAux = std::make_shared<edm::RunAuxiliary>(aux.run(), aux.time(), aux.time());
      auto rp = std::make_shared<edm::RunPrincipal>(runAux, m_->reg(), m_->pc_, nullptr, 0);
      auto lbp = std::make_shared<edm::LuminosityBlockPrincipal>(m_->reg(), m_->pc_, nullptr, 0);
      lbp->setAux(edm::LuminosityBlockAuxiliary(rp->run(), 1, aux.time(), aux.time()));
      auto history = m_->phreg_->getMapped(eaux->processHistoryID());
      m_->ep_->fillEventPrincipal(*eaux,
                                  history,
                                  std::move(eventSelectionIDs),
                                  std::move(branchListIndexes),
                                  *(m_->provRetriever_),
                                  m_->reader_.get());
      lbp->setRunPrincipal(rp);
      m_->ep_->setLuminosityBlockPrincipal(lbp.get());
      m_->processNames_ = m_->ep_->processHistory();

      edm::Event event(*m_->ep_, m_->md_, nullptr);

      //Make the event principal accessible to edm::Ref's
      Operate sentry(m_->ep_->prodGetter());
      process(event);
    } catch (std::exception const& iEx) {
      std::cout << "While processing entry " << iEntry << " the following exception was caught \n"
                << iEx.what() << std::endl;
    } catch (...) {
      std::cout << "While processing entry " << iEntry << " an unknown exception was caught" << std::endl;
    }
  }
  return everythingOK_ ? kTRUE : kFALSE;
}

void TFWLiteSelectorBasic::SlaveTerminate() { postProcessing(*fOutput); }

void TFWLiteSelectorBasic::Terminate() { terminate(*fOutput); }

void TFWLiteSelectorBasic::setupNewFile(TFile& iFile) {
  //look up meta-data
  //get product registry

  //std::vector<edm::EventProcessHistoryID> eventProcessHistoryIDs_;
  TTree* metaDataTree = dynamic_cast<TTree*>(iFile.Get(edm::poolNames::metaDataTreeName().c_str()));
  if (!metaDataTree) {
    std::cout << "could not find TTree " << edm::poolNames::metaDataTreeName() << std::endl;
    everythingOK_ = false;
    return;
  }
  edm::FileFormatVersion* fftPtr = &(m_->fileFormatVersion_);
  if (metaDataTree->FindBranch(edm::poolNames::fileFormatVersionBranchName().c_str()) != nullptr) {
    metaDataTree->SetBranchAddress(edm::poolNames::fileFormatVersionBranchName().c_str(), &fftPtr);
  }

  edm::ProductRegistry* pReg = &(*m_->reg_);
  metaDataTree->SetBranchAddress(edm::poolNames::productDescriptionBranchName().c_str(), &(pReg));

  typedef std::map<edm::ParameterSetID, edm::ParameterSetBlob> PsetMap;
  PsetMap psetMap;
  PsetMap* psetMapPtr = &psetMap;
  if (metaDataTree->FindBranch(edm::poolNames::parameterSetMapBranchName().c_str()) != nullptr) {
    metaDataTree->SetBranchAddress(edm::poolNames::parameterSetMapBranchName().c_str(), &psetMapPtr);
  } else {
    TTree* psetTree = dynamic_cast<TTree*>(iFile.Get(edm::poolNames::parameterSetsTreeName().c_str()));
    if (nullptr == psetTree) {
      throw edm::Exception(edm::errors::FileReadError)
          << "Could not find tree " << edm::poolNames::parameterSetsTreeName() << " in the input file.\n";
    }
    typedef std::pair<edm::ParameterSetID, edm::ParameterSetBlob> IdToBlobs;
    IdToBlobs idToBlob;
    IdToBlobs* pIdToBlob = &idToBlob;
    psetTree->SetBranchAddress(edm::poolNames::idToParameterSetBlobsBranchName().c_str(), &pIdToBlob);
    for (long long i = 0; i != psetTree->GetEntries(); ++i) {
      psetTree->GetEntry(i);
      psetMap.insert(idToBlob);
    }
  }

  edm::ProcessHistoryRegistry::vector_type pHistVector;
  edm::ProcessHistoryRegistry::vector_type* pHistVectorPtr = &pHistVector;
  if (metaDataTree->FindBranch(edm::poolNames::processHistoryBranchName().c_str()) != nullptr) {
    metaDataTree->SetBranchAddress(edm::poolNames::processHistoryBranchName().c_str(), &pHistVectorPtr);
  }

  edm::ProcessConfigurationVector procConfigVector;
  edm::ProcessConfigurationVector* procConfigVectorPtr = &procConfigVector;
  if (metaDataTree->FindBranch(edm::poolNames::processConfigurationBranchName().c_str()) != nullptr) {
    metaDataTree->SetBranchAddress(edm::poolNames::processConfigurationBranchName().c_str(), &procConfigVectorPtr);
  }

  auto branchIDListsHelper = std::make_shared<edm::BranchIDListHelper>();
  edm::BranchIDLists const* branchIDListsPtr = &branchIDListsHelper->branchIDLists();
  if (metaDataTree->FindBranch(edm::poolNames::branchIDListBranchName().c_str()) != nullptr) {
    metaDataTree->SetBranchAddress(edm::poolNames::branchIDListBranchName().c_str(), &branchIDListsPtr);
  }

  metaDataTree->GetEntry(0);

  for (auto& prod : m_->reg_->productListUpdator()) {
    prod.second.init();
    setIsMergeable(prod.second);
  }

  // Merge into the registries. For now, we do NOT merge the product registry.
  edm::pset::Registry& psetRegistry = *edm::pset::Registry::instance();
  for (auto const& entry : psetMap) {
    edm::ParameterSet pset(entry.second.pset());
    pset.setID(entry.first);
    psetRegistry.insertMapped(pset);
  }

  for (auto const& ph : pHistVector) {
    m_->phreg_->registerProcessHistory(ph);
  }

  m_->pointerToBranchBuffer_.erase(m_->pointerToBranchBuffer_.begin(), m_->pointerToBranchBuffer_.end());

  std::unique_ptr<edm::ProductRegistry> newReg = std::make_unique<edm::ProductRegistry>();

  edm::ProductRegistry::ProductList& prodList = m_->reg_->productListUpdator();
  {
    for (auto& item : prodList) {
      edm::BranchDescription& prod = item.second;
      //std::cout << "productname = " << item.second << " end " << std::endl;
      std::string newFriendlyName = edm::friendlyname::friendlyName(prod.className());
      if (newFriendlyName == prod.friendlyClassName()) {
        newReg->copyProduct(prod);
      } else {
        if (m_->fileFormatVersion_.splitProductIDs()) {
          throw edm::Exception(edm::errors::UnimplementedFeature)
              << "Cannot change friendly class name algorithm without more development work\n"
              << "to update BranchIDLists.  Contact the framework group.\n";
        }
        edm::BranchDescription newBD(prod);
        newBD.updateFriendlyClassName();
        newReg->copyProduct(newBD);
      }
      prod.init();
    }

    m_->reg().reset(newReg.release());
  }

  edm::ProductRegistry::ProductList& prodList2 = m_->reg_->productListUpdator();
  std::vector<edm::EventEntryDescription> temp(prodList2.size(), edm::EventEntryDescription());
  m_->prov_.swap(temp);
  m_->pointerToBranchBuffer_.reserve(prodList2.size());

  for (auto& item : prodList2) {
    edm::BranchDescription& prod = item.second;
    if (prod.branchType() == edm::InEvent) {
      prod.init();
      //NEED to do this and check to see if branch exists
      if (m_->tree_->GetBranch(prod.branchName().c_str()) == nullptr) {
        prod.setDropped(true);
      }

      //std::cout << "id " << it->first << " branch " << it->second << std::endl;
      //m_->pointerToBranchBuffer_.push_back(&(*itB));
      //void* tmp = &(m_->pointerToBranchBuffer_.back());
      //edm::EventEntryDescription* tmp = &(*itB);
      //CDJ need to fix provenance and be backwards compatible, for now just don't read the branch
      //m_->metaTree_->SetBranchAddress(prod.branchName().c_str(), tmp);
    }
  }
  m_->branchIDListHelper_->updateFromInput(*branchIDListsPtr);
  m_->reg_->setFrozen();
  m_->bidToDesc_->clear();
  for (auto const& p : m_->reg_->productList()) {
    m_->bidToDesc_->emplace(p.second.branchID().id(), &p.second);
  }
  m_->ep_ = std::make_shared<edm::EventPrincipal>(
      m_->reg(), m_->branchIDListHelper(), m_->thinnedAssociationsHelper(), m_->pc_, nullptr);
  everythingOK_ = true;
}

//
// const member functions
//

//
// static member functions
//
