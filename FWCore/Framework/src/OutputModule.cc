/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/OutputModule.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/insertSelectedProcesses.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/OutputModuleDescription.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"
#include "FWCore/Framework/interface/PrincipalGetAdapter.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "SharedResourcesRegistry.h"

#include <cassert>
#include <iostream>

namespace edm {

  // -------------------------------------------------------
  OutputModule::OutputModule(ParameterSet const& pset) :
    maxEvents_(-1),
    remainingEvents_(maxEvents_),
    keptProducts_(),
    hasNewlyDroppedBranch_(),
    process_name_(),
    productSelectorRules_(pset, "outputCommands", "OutputModule"),
    productSelector_(),
    moduleDescription_(),
    wantAllEvents_(false),
    selectors_(),
    selector_config_id_(),
    droppedBranchIDToKeptBranchID_(),
    branchIDLists_(new BranchIDLists),
    origBranchIDLists_(nullptr),
    thinnedAssociationsHelper_(new ThinnedAssociationsHelper) {

    hasNewlyDroppedBranch_.fill(false);

    Service<service::TriggerNamesService> tns;
    process_name_ = tns->getProcessName();

    selectEvents_ =
      pset.getUntrackedParameterSet("SelectEvents", ParameterSet());

    selectEvents_.registerIt(); // Just in case this PSet is not registered

    selector_config_id_ = selectEvents_.id();
    selectors_.resize(1);
    //need to set wantAllEvents_ in constructor
    // we will make the remaining selectors once we know how many streams
    wantAllEvents_ = detail::configureEventSelector(selectEvents_,
                                                          process_name_,
                                                          getAllTriggerNames(),
                                                          selectors_[0],
                                                          consumesCollector());

    SharedResourcesRegistry::instance()->registerSharedResource(
                                                                SharedResourcesRegistry::kLegacyModuleResourceName);

  }

  void OutputModule::configure(OutputModuleDescription const& desc) {
    remainingEvents_ = maxEvents_ = desc.maxEvents_;
    origBranchIDLists_ = desc.branchIDLists_;
  }

  void OutputModule::selectProducts(ProductRegistry const& preg,
                                    ThinnedAssociationsHelper const& thinnedAssociationsHelper) {
    if(productSelector_.initialized()) return;
    productSelector_.initialize(productSelectorRules_, preg.allBranchDescriptions());

    // TODO: See if we can collapse keptProducts_ and productSelector_ into a
    // single object. See the notes in the header for ProductSelector
    // for more information.

    std::map<BranchID, BranchDescription const*> trueBranchIDToKeptBranchDesc;
    std::vector<BranchDescription const*> associationDescriptions;
    std::set<BranchID> keptProductsInEvent;
    std::set<std::string> processesWithSelectedMergeableRunProducts;

    for(auto const& it : preg.productList()) {
      BranchDescription const& desc = it.second;
      if(desc.transient()) {
        // if the class of the branch is marked transient, output nothing
      } else if(!desc.present() && !desc.produced()) {
        // else if the branch containing the product has been previously dropped,
        // output nothing
      } else if(desc.unwrappedType() == typeid(ThinnedAssociation)) {
        associationDescriptions.push_back(&desc);
      } else if(selected(desc)) {
        keepThisBranch(desc, trueBranchIDToKeptBranchDesc, keptProductsInEvent);
        insertSelectedProcesses(desc,
                                processesWithSelectedMergeableRunProducts);
      } else {
        // otherwise, output nothing,
        // and mark the fact that there is a newly dropped branch of this type.
        hasNewlyDroppedBranch_[desc.branchType()] = true;
      }
    }

    setProcessesWithSelectedMergeableRunProducts(processesWithSelectedMergeableRunProducts);

    thinnedAssociationsHelper.selectAssociationProducts(associationDescriptions,
                                                        keptProductsInEvent,
                                                        keepAssociation_);

    for(auto association : associationDescriptions) {
      if(keepAssociation_[association->branchID()]) {
        keepThisBranch(*association, trueBranchIDToKeptBranchDesc, keptProductsInEvent);
      } else {
        hasNewlyDroppedBranch_[association->branchType()] = true;
      }
    }

    // Now fill in a mapping needed in the case that a branch was dropped while its EDAlias was kept.
    ProductSelector::fillDroppedToKept(preg, trueBranchIDToKeptBranchDesc, droppedBranchIDToKeptBranchID_);

    thinnedAssociationsHelper_->updateFromParentProcess(thinnedAssociationsHelper, keepAssociation_, droppedBranchIDToKeptBranchID_);
  }

  void OutputModule::keepThisBranch(BranchDescription const& desc,
                                    std::map<BranchID, BranchDescription const*>& trueBranchIDToKeptBranchDesc,
                                    std::set<BranchID>& keptProductsInEvent) {

    ProductSelector::checkForDuplicateKeptBranch(desc,
                                                 trueBranchIDToKeptBranchDesc);

    EDGetToken token;

    std::vector<std::string> missingDictionaries;
    if (!checkDictionary(missingDictionaries, desc.className(), desc.unwrappedType())) {
      std::string context("Calling OutputModule::keepThisBranch, checking dictionaries for kept types");
      throwMissingDictionariesException(missingDictionaries, context);
    }

    switch (desc.branchType()) {
    case InEvent:
      {
        if(desc.produced()) {
          keptProductsInEvent.insert(desc.originalBranchID());
        } else {
          keptProductsInEvent.insert(desc.branchID());
        }
        token = consumes(TypeToGet{desc.unwrappedTypeID(),PRODUCT_TYPE},
                     InputTag{desc.moduleLabel(),
                     desc.productInstanceName(),
                     desc.processName()});
        break;
      }
    case InLumi:
      {
        token = consumes<InLumi>(TypeToGet{desc.unwrappedTypeID(),PRODUCT_TYPE},
                                  InputTag(desc.moduleLabel(),
                                  desc.productInstanceName(),
                                  desc.processName()));
        break;
      }
    case InRun:
      {
        token = consumes<InRun>(TypeToGet{desc.unwrappedTypeID(),PRODUCT_TYPE},
                                 InputTag(desc.moduleLabel(),
                                 desc.productInstanceName(),
                                 desc.processName()));
        break;
      }
    default:
      assert(false);
      break;
    }
    // Now put it in the list of selected branches.
    keptProducts_[desc.branchType()].push_back(std::make_pair(&desc, token));
  }

  OutputModule::~OutputModule() { }

  void OutputModule::doPreallocate(PreallocationConfiguration const& iPC) {
    auto nstreams = iPC.numberOfStreams();
    selectors_.resize(nstreams);

    bool seenFirst = false;
    for(auto& s : selectors_) {
      if(seenFirst) {
        detail::configureEventSelector(selectEvents_,
                                       process_name_,
                                       getAllTriggerNames(),
                                       s,
                                       consumesCollector());
      }
      seenFirst = true;
    }
  }

  
  void OutputModule::doBeginJob() {
    std::vector<std::string> res = {SharedResourcesRegistry::kLegacyModuleResourceName};
    resourceAcquirer_ = SharedResourcesRegistry::instance()->createAcquirer(res);

    this->beginJob();
  }

  void OutputModule::doEndJob() {
    endJob();
  }


  Trig OutputModule::getTriggerResults(EDGetTokenT<TriggerResults> const& token, EventForOutput const& e) const {
    //This cast is safe since we only call const functions of the EventForOutputafter this point
    Trig result;
    e.getByToken<TriggerResults>(token, result);
    return result;
  }
  
  bool OutputModule::needToRunSelection() const {
    return !wantAllEvents_;
  }
  
  std::vector<ProductResolverIndexAndSkipBit>
  OutputModule::productsUsedBySelection() const {
    std::vector<ProductResolverIndexAndSkipBit> returnValue;
    auto const& s = selectors_[0];
    auto const n = s.numberOfTokens();
    returnValue.reserve(n);
    
    for(unsigned int i=0; i< n;++i) {
      returnValue.emplace_back(uncheckedIndexFrom(s.token(i)));
    }
    return returnValue;
  }

  bool OutputModule::prePrefetchSelection(StreamID id, EventPrincipal const& ep, ModuleCallingContext const* mcc) {
    if(wantAllEvents_) return true;
    auto& s = selectors_[id.value()];
    EventForOutput e(ep, moduleDescription_, mcc);
    e.setConsumer(this);
    return s.wantEvent(e);
  }

  bool
  OutputModule::doEvent(EventPrincipal const& ep,
                        EventSetup const&,
                        ActivityRegistry* act,
                        ModuleCallingContext const* mcc) {

    FDEBUG(2) << "writeEvent called\n";

    {
      EventForOutput e(ep, moduleDescription_, mcc);
      e.setConsumer(this);
      EventSignalsSentry sentry(act,mcc);
      write(e);
    }
    if(remainingEvents_ > 0) {
      --remainingEvents_;
    }
    return true;
  }

//   bool OutputModule::wantEvent(EventForOutput const& ev)
//   {
//     getTriggerResults(ev);
//     bool eventAccepted = false;

//     typedef std::vector<NamedEventSelector>::const_iterator iter;
//     for(iter i = selectResult_.begin(), e = selectResult_.end();
//          !eventAccepted && i != e; ++i)
//       {
//         eventAccepted = i->acceptEvent(*prods_);
//       }

//     FDEBUG(2) << "Accept event " << ep.id() << " " << eventAccepted << "\n";
//     return eventAccepted;
//   }

  bool
  OutputModule::doBeginRun(RunPrincipal const& rp,
                           EventSetup const&,
                           ModuleCallingContext const* mcc) {
    FDEBUG(2) << "beginRun called\n";
    RunForOutput r(rp, moduleDescription_, mcc,false);
    r.setConsumer(this);
    beginRun(r);
    return true;
  }

  bool
  OutputModule::doEndRun(RunPrincipal const& rp,
                         EventSetup const&,
                         ModuleCallingContext const* mcc) {
    FDEBUG(2) << "endRun called\n";
    RunForOutput r(rp, moduleDescription_, mcc,true);
    r.setConsumer(this);
    endRun(r);
    return true;
  }

  void
  OutputModule::doWriteRun(RunPrincipal const& rp,
                           ModuleCallingContext const* mcc,
                           MergeableRunProductMetadata const* mrpm) {
    FDEBUG(2) << "writeRun called\n";
    RunForOutput r(rp, moduleDescription_, mcc,true, mrpm);
    r.setConsumer(this);
    writeRun(r);
  }

  bool
  OutputModule::doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                       EventSetup const&,
                                       ModuleCallingContext const* mcc) {
    FDEBUG(2) << "beginLuminosityBlock called\n";
    LuminosityBlockForOutput lb(lbp, moduleDescription_, mcc,false);
    lb.setConsumer(this);
    beginLuminosityBlock(lb);
    return true;
  }

  bool
  OutputModule::doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                     EventSetup const&,
                                     ModuleCallingContext const* mcc) {
    FDEBUG(2) << "endLuminosityBlock called\n";
    LuminosityBlockForOutput lb(lbp, moduleDescription_, mcc,true);
    lb.setConsumer(this);
    endLuminosityBlock(lb);
    return true;
  }

  void OutputModule::doWriteLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                            ModuleCallingContext const* mcc) {
    FDEBUG(2) << "writeLuminosityBlock called\n";
    LuminosityBlockForOutput lb(lbp, moduleDescription_, mcc,true);
    lb.setConsumer(this);
    writeLuminosityBlock(lb);
  }

  void OutputModule::doOpenFile(FileBlock const& fb) {
    openFile(fb);
  }

  void OutputModule::doRespondToOpenInputFile(FileBlock const& fb) {
    respondToOpenInputFile(fb);
  }

  void OutputModule::doRespondToCloseInputFile(FileBlock const& fb) {
    respondToCloseInputFile(fb);
  }

  void OutputModule::doCloseFile() {
    if(isFileOpen()) {
      reallyCloseFile();
    }
  }

  void OutputModule::reallyCloseFile() {
  }

  BranchIDLists const*
  OutputModule::branchIDLists() {
    if(!droppedBranchIDToKeptBranchID_.empty()) {
      // Make a private copy of the BranchIDLists.
      *branchIDLists_ = *origBranchIDLists_;
      // Check for branches dropped while an EDAlias was kept.
      for(BranchIDList& branchIDList : *branchIDLists_) {
        for(BranchID::value_type& branchID : branchIDList) {
          // Replace BranchID of each dropped branch with that of the kept alias, so the alias branch will have the product ID of the original branch.
          std::map<BranchID::value_type, BranchID::value_type>::const_iterator iter = droppedBranchIDToKeptBranchID_.find(branchID);
          if(iter != droppedBranchIDToKeptBranchID_.end()) {
            branchID = iter->second;
          }
        }
      }
      return branchIDLists_.get();
    }
    return origBranchIDLists_;
  }

  ThinnedAssociationsHelper const*
  OutputModule::thinnedAssociationsHelper() const {
    return thinnedAssociationsHelper_.get();
  }

  ModuleDescription const&
  OutputModule::description() const {
    return moduleDescription_;
  }

  bool
  OutputModule::selected(BranchDescription const& desc) const {
    return productSelector_.selected(desc);
  }

  void
  OutputModule::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
  }
  
  void
  OutputModule::fillDescription(ParameterSetDescription& desc, std::vector<std::string> const& defaultOutputCommands) {
    ProductSelectorRules::fillDescription(desc, "outputCommands",defaultOutputCommands);
    EventSelector::fillDescription(desc);
  }
  
  void
  OutputModule::prevalidate(ConfigurationDescriptions& ) {
  }
  

  static const std::string kBaseType("OutputModule");
  const std::string&
  OutputModule::baseType() {
    return kBaseType;
  }
  
  void
  OutputModule::setEventSelectionInfo(std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
                                      bool anyProductProduced) {
    selector_config_id_ = detail::registerProperSelectionInfo(getParameterSet(selector_config_id_),
                                      description().moduleLabel(),
                                                      outputModulePathPositions,
                                                      anyProductProduced);
  }
}
