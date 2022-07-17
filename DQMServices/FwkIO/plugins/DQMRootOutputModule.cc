// -*- C++ -*-
//
// Package:     FwkIO
// Class  :     DQMRootOutputModule
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Fri Apr 29 13:26:29 CDT 2011
//

// system include files
#include <algorithm>
#include <iostream>
#include <memory>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"

#include "oneapi/tbb/task_arena.h"

// user include files
#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"

#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include "DataFormats/Histograms/interface/DQMToken.h"

#include "format.h"

namespace {
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

  class TreeHelperBase {
  public:
    TreeHelperBase() : m_wasFilled(false), m_firstIndex(0), m_lastIndex(0) {}
    virtual ~TreeHelperBase() {}
    void fill(MonitorElement* iElement) {
      doFill(iElement);
      if (m_wasFilled) {
        ++m_lastIndex;
      }
      m_wasFilled = true;
    }
    bool wasFilled() const { return m_wasFilled; }
    void getRangeAndReset(ULong64_t& iFirstIndex, ULong64_t& iLastIndex) {
      iFirstIndex = m_firstIndex;
      iLastIndex = m_lastIndex;
      m_wasFilled = false;
      m_firstIndex = m_lastIndex + 1;
      m_lastIndex = m_firstIndex;
    }

  private:
    virtual void doFill(MonitorElement*) = 0;
    bool m_wasFilled;
    ULong64_t m_firstIndex;
    ULong64_t m_lastIndex;
  };

  template <class T>
  class TreeHelper : public TreeHelperBase {
  public:
    TreeHelper(TTree* iTree, std::string* iFullNameBufferPtr)
        : m_tree(iTree), m_flagBuffer(0), m_fullNameBufferPtr(iFullNameBufferPtr) {
      setup();
    }
    void doFill(MonitorElement* iElement) override {
      *m_fullNameBufferPtr = iElement->getFullname();
      m_flagBuffer = 0;
      m_bufferPtr = dynamic_cast<T*>(iElement->getRootObject());
      assert(nullptr != m_bufferPtr);
      //std::cout <<"#entries: "<<m_bufferPtr->GetEntries()<<std::endl;
      tbb::this_task_arena::isolate([&] { m_tree->Fill(); });
    }

  private:
    void setup() {
      m_tree->Branch(kFullNameBranch, &m_fullNameBufferPtr);
      m_tree->Branch(kFlagBranch, &m_flagBuffer);

      m_bufferPtr = nullptr;
      m_tree->Branch(kValueBranch, &m_bufferPtr, 128 * 1024, 0);
    }
    TTree* m_tree;
    uint32_t m_flagBuffer;
    std::string* m_fullNameBufferPtr;
    T* m_bufferPtr;
  };

  class IntTreeHelper : public TreeHelperBase {
  public:
    IntTreeHelper(TTree* iTree, std::string* iFullNameBufferPtr)
        : m_tree(iTree), m_flagBuffer(0), m_fullNameBufferPtr(iFullNameBufferPtr) {
      setup();
    }

    void doFill(MonitorElement* iElement) override {
      *m_fullNameBufferPtr = iElement->getFullname();
      m_flagBuffer = 0;
      m_buffer = iElement->getIntValue();
      tbb::this_task_arena::isolate([&] { m_tree->Fill(); });
    }

  private:
    void setup() {
      m_tree->Branch(kFullNameBranch, &m_fullNameBufferPtr);
      m_tree->Branch(kFlagBranch, &m_flagBuffer);
      m_tree->Branch(kValueBranch, &m_buffer);
    }
    TTree* m_tree;
    uint32_t m_flagBuffer;
    std::string* m_fullNameBufferPtr;
    Long64_t m_buffer;
  };

  class FloatTreeHelper : public TreeHelperBase {
  public:
    FloatTreeHelper(TTree* iTree, std::string* iFullNameBufferPtr)
        : m_tree(iTree), m_flagBuffer(0), m_fullNameBufferPtr(iFullNameBufferPtr) {
      setup();
    }
    void doFill(MonitorElement* iElement) override {
      *m_fullNameBufferPtr = iElement->getFullname();
      m_flagBuffer = 0;
      m_buffer = iElement->getFloatValue();
      tbb::this_task_arena::isolate([&] { m_tree->Fill(); });
    }

  private:
    void setup() {
      m_tree->Branch(kFullNameBranch, &m_fullNameBufferPtr);
      m_tree->Branch(kFlagBranch, &m_flagBuffer);
      m_tree->Branch(kValueBranch, &m_buffer);
    }

    TTree* m_tree;
    uint32_t m_flagBuffer;
    std::string* m_fullNameBufferPtr;
    double m_buffer;
  };

  class StringTreeHelper : public TreeHelperBase {
  public:
    StringTreeHelper(TTree* iTree, std::string* iFullNameBufferPtr)
        : m_tree(iTree), m_flagBuffer(0), m_fullNameBufferPtr(iFullNameBufferPtr), m_bufferPtr(&m_buffer) {
      setup();
    }
    void doFill(MonitorElement* iElement) override {
      *m_fullNameBufferPtr = iElement->getFullname();
      m_flagBuffer = 0;
      m_buffer = iElement->getStringValue();
      tbb::this_task_arena::isolate([&] { m_tree->Fill(); });
    }

  private:
    void setup() {
      m_tree->Branch(kFullNameBranch, &m_fullNameBufferPtr);
      m_tree->Branch(kFlagBranch, &m_flagBuffer);
      m_tree->Branch(kValueBranch, &m_bufferPtr);
    }

    TTree* m_tree;
    uint32_t m_flagBuffer;
    std::string* m_fullNameBufferPtr;
    std::string m_buffer;
    std::string* m_bufferPtr;
  };

}  // namespace

namespace edm {
  class ModuleCallingContext;
}

class DQMRootOutputModule : public edm::one::OutputModule<> {
public:
  explicit DQMRootOutputModule(edm::ParameterSet const& pset);
  void beginJob() override;
  ~DQMRootOutputModule() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void write(edm::EventForOutput const& e) override;
  void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override;
  void writeRun(edm::RunForOutput const&) override;
  bool isFileOpen() const override;
  void openFile(edm::FileBlock const&) override;
  void reallyCloseFile() override;

  void startEndFile();
  void finishEndFile();
  std::string m_fileName;
  std::string m_logicalFileName;
  std::unique_ptr<TFile> m_file;
  std::vector<std::shared_ptr<TreeHelperBase> > m_treeHelpers;

  unsigned int m_run;
  unsigned int m_lumi;
  unsigned int m_type;
  unsigned int m_presentHistoryIndex;
  ULong64_t m_beginTime;
  ULong64_t m_endTime;
  ULong64_t m_firstIndex;
  ULong64_t m_lastIndex;
  unsigned int m_filterOnRun;

  std::string m_fullNameBuffer;
  std::string* m_fullNameBufferPtr;
  std::map<unsigned int, unsigned int> m_dqmKindToTypeIndex;
  TTree* m_indicesTree;

  std::vector<edm::ProcessHistoryID> m_seenHistories;
  edm::ProcessHistoryRegistry m_processHistoryRegistry;
  edm::JobReport::Token m_jrToken;
};

//
// constants, enums and typedefs
//

static TreeHelperBase* makeHelper(unsigned int iTypeIndex, TTree* iTree, std::string* iFullNameBufferPtr) {
  switch (iTypeIndex) {
    case kIntIndex:
      return new IntTreeHelper(iTree, iFullNameBufferPtr);
    case kFloatIndex:
      return new FloatTreeHelper(iTree, iFullNameBufferPtr);
    case kStringIndex:
      return new StringTreeHelper(iTree, iFullNameBufferPtr);
    case kTH1FIndex:
      return new TreeHelper<TH1F>(iTree, iFullNameBufferPtr);
    case kTH1SIndex:
      return new TreeHelper<TH1S>(iTree, iFullNameBufferPtr);
    case kTH1DIndex:
      return new TreeHelper<TH1D>(iTree, iFullNameBufferPtr);
    case kTH1IIndex:
      return new TreeHelper<TH1I>(iTree, iFullNameBufferPtr);
    case kTH2FIndex:
      return new TreeHelper<TH2F>(iTree, iFullNameBufferPtr);
    case kTH2SIndex:
      return new TreeHelper<TH2S>(iTree, iFullNameBufferPtr);
    case kTH2DIndex:
      return new TreeHelper<TH2D>(iTree, iFullNameBufferPtr);
    case kTH2IIndex:
      return new TreeHelper<TH2I>(iTree, iFullNameBufferPtr);
    case kTH3FIndex:
      return new TreeHelper<TH3F>(iTree, iFullNameBufferPtr);
    case kTProfileIndex:
      return new TreeHelper<TProfile>(iTree, iFullNameBufferPtr);
    case kTProfile2DIndex:
      return new TreeHelper<TProfile2D>(iTree, iFullNameBufferPtr);
  }
  assert(false);
  return nullptr;
}

//
// static data member definitions
//

//
// constructors and destructor
//
DQMRootOutputModule::DQMRootOutputModule(edm::ParameterSet const& pset)
    : edm::one::OutputModuleBase::OutputModuleBase(pset),
      edm::one::OutputModule<>(pset),
      m_fileName(pset.getUntrackedParameter<std::string>("fileName")),
      m_logicalFileName(pset.getUntrackedParameter<std::string>("logicalFileName")),
      m_file(nullptr),
      m_treeHelpers(kNIndicies, std::shared_ptr<TreeHelperBase>()),
      m_presentHistoryIndex(0),
      m_filterOnRun(pset.getUntrackedParameter<unsigned int>("filterOnRun")),
      m_fullNameBufferPtr(&m_fullNameBuffer),
      m_indicesTree(nullptr) {
  // Declare dependencies for all Lumi and Run tokens here. In
  // principle could use the keep statements, but then DQMToken would
  // have to be made persistent (transient products are ignored),
  // which would lead to a need to (finally) remove underscores from
  // DQM module labels.
  consumesMany<DQMToken, edm::InLumi>();
  consumesMany<DQMToken, edm::InRun>();
}

// DQMRootOutputModule::DQMRootOutputModule(const DQMRootOutputModule& rhs)
// {
//    // do actual copying here;
// }

void DQMRootOutputModule::beginJob() {}

DQMRootOutputModule::~DQMRootOutputModule() {}

//
// assignment operators
//
// const DQMRootOutputModule& DQMRootOutputModule::operator=(const DQMRootOutputModule& rhs)
// {
//   //An exception safe implementation is
//   DQMRootOutputModule temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
bool DQMRootOutputModule::isFileOpen() const { return nullptr != m_file.get(); }

void DQMRootOutputModule::openFile(edm::FileBlock const&) {
  //NOTE: I need to also set the I/O performance settings

  m_file = std::make_unique<TFile>(m_fileName.c_str(),
                                   "RECREATE",
                                   "1"  //This is the file format version number
  );

  edm::Service<edm::JobReport> jr;
  cms::Digest branchHash;
  std::string guid{edm::createGlobalIdentifier()};
  std::transform(guid.begin(), guid.end(), guid.begin(), (int (*)(int))std::toupper);

  m_file->WriteObject(&guid, kCmsGuid);
  m_jrToken = jr->outputFileOpened(m_fileName,
                                   m_logicalFileName,
                                   std::string(),
                                   "DQMRootOutputModule",
                                   description().moduleLabel(),
                                   guid,
                                   std::string(),
                                   branchHash.digest().toString(),
                                   std::vector<std::string>());

  m_indicesTree = new TTree(kIndicesTree, kIndicesTree);
  m_indicesTree->Branch(kRunBranch, &m_run);
  m_indicesTree->Branch(kLumiBranch, &m_lumi);
  m_indicesTree->Branch(kProcessHistoryIndexBranch, &m_presentHistoryIndex);
  m_indicesTree->Branch(kBeginTimeBranch, &m_beginTime);
  m_indicesTree->Branch(kEndTimeBranch, &m_endTime);
  m_indicesTree->Branch(kTypeBranch, &m_type);
  m_indicesTree->Branch(kFirstIndex, &m_firstIndex);
  m_indicesTree->Branch(kLastIndex, &m_lastIndex);
  m_indicesTree->SetDirectory(m_file.get());

  unsigned int i = 0;
  for (std::vector<std::shared_ptr<TreeHelperBase> >::iterator it = m_treeHelpers.begin(), itEnd = m_treeHelpers.end();
       it != itEnd;
       ++it, ++i) {
    //std::cout <<"making "<<kTypeNames[i]<<std::endl;
    TTree* tree = new TTree(kTypeNames[i], kTypeNames[i]);
    *it = std::shared_ptr<TreeHelperBase>(makeHelper(i, tree, m_fullNameBufferPtr));
    tree->SetDirectory(m_file.get());  //TFile takes ownership
  }

  m_dqmKindToTypeIndex[(int)MonitorElement::Kind::INT] = kIntIndex;
  m_dqmKindToTypeIndex[(int)MonitorElement::Kind::REAL] = kFloatIndex;
  m_dqmKindToTypeIndex[(int)MonitorElement::Kind::STRING] = kStringIndex;
  m_dqmKindToTypeIndex[(int)MonitorElement::Kind::TH1F] = kTH1FIndex;
  m_dqmKindToTypeIndex[(int)MonitorElement::Kind::TH1S] = kTH1SIndex;
  m_dqmKindToTypeIndex[(int)MonitorElement::Kind::TH1D] = kTH1DIndex;
  m_dqmKindToTypeIndex[(int)MonitorElement::Kind::TH1I] = kTH1IIndex;
  m_dqmKindToTypeIndex[(int)MonitorElement::Kind::TH2F] = kTH2FIndex;
  m_dqmKindToTypeIndex[(int)MonitorElement::Kind::TH2S] = kTH2SIndex;
  m_dqmKindToTypeIndex[(int)MonitorElement::Kind::TH2D] = kTH2DIndex;
  m_dqmKindToTypeIndex[(int)MonitorElement::Kind::TH2I] = kTH2IIndex;
  m_dqmKindToTypeIndex[(int)MonitorElement::Kind::TH3F] = kTH3FIndex;
  m_dqmKindToTypeIndex[(int)MonitorElement::Kind::TPROFILE] = kTProfileIndex;
  m_dqmKindToTypeIndex[(int)MonitorElement::Kind::TPROFILE2D] = kTProfile2DIndex;
}

void DQMRootOutputModule::write(edm::EventForOutput const&) {}

void DQMRootOutputModule::writeLuminosityBlock(edm::LuminosityBlockForOutput const& iLumi) {
  //std::cout << "DQMRootOutputModule::writeLuminosityBlock"<< std::endl;
  edm::Service<DQMStore> dstore;
  m_run = iLumi.id().run();
  m_lumi = iLumi.id().value();
  m_beginTime = iLumi.beginTime().value();
  m_endTime = iLumi.endTime().value();
  bool shouldWrite = (m_filterOnRun == 0 || (m_filterOnRun != 0 && m_filterOnRun == m_run));

  if (!shouldWrite)
    return;
  std::vector<MonitorElement*> items(dstore->getAllContents("", m_run, m_lumi));
  for (std::vector<MonitorElement*>::iterator it = items.begin(), itEnd = items.end(); it != itEnd; ++it) {
    assert((*it)->getScope() == MonitorElementData::Scope::LUMI);
    std::map<unsigned int, unsigned int>::iterator itFound = m_dqmKindToTypeIndex.find((int)(*it)->kind());
    assert(itFound != m_dqmKindToTypeIndex.end());
    m_treeHelpers[itFound->second]->fill(*it);
  }

  const edm::ProcessHistoryID& id = iLumi.processHistoryID();
  std::vector<edm::ProcessHistoryID>::iterator itFind = std::find(m_seenHistories.begin(), m_seenHistories.end(), id);
  if (itFind == m_seenHistories.end()) {
    m_processHistoryRegistry.registerProcessHistory(iLumi.processHistory());
    m_presentHistoryIndex = m_seenHistories.size();
    m_seenHistories.push_back(id);
  } else {
    m_presentHistoryIndex = itFind - m_seenHistories.begin();
  }

  //Now store the relationship between run/lumi and indices in the other TTrees
  bool storedLumiIndex = false;
  unsigned int typeIndex = 0;
  for (std::vector<std::shared_ptr<TreeHelperBase> >::iterator it = m_treeHelpers.begin(), itEnd = m_treeHelpers.end();
       it != itEnd;
       ++it, ++typeIndex) {
    if ((*it)->wasFilled()) {
      m_type = typeIndex;
      (*it)->getRangeAndReset(m_firstIndex, m_lastIndex);
      storedLumiIndex = true;
      tbb::this_task_arena::isolate([&] { m_indicesTree->Fill(); });
    }
  }
  if (not storedLumiIndex) {
    //need to record lumis even if we stored no MonitorElements since some later DQM modules
    // look to see what lumis were processed
    m_type = kNoTypesStored;
    m_firstIndex = 0;
    m_lastIndex = 0;
    tbb::this_task_arena::isolate([&] { m_indicesTree->Fill(); });
  }

  edm::Service<edm::JobReport> jr;
  jr->reportLumiSection(m_jrToken, m_run, m_lumi);
}

void DQMRootOutputModule::writeRun(edm::RunForOutput const& iRun) {
  //std::cout << "DQMRootOutputModule::writeRun"<< std::endl;
  edm::Service<DQMStore> dstore;
  m_run = iRun.id().run();
  m_lumi = 0;
  m_beginTime = iRun.beginTime().value();
  m_endTime = iRun.endTime().value();
  bool shouldWrite = (m_filterOnRun == 0 || (m_filterOnRun != 0 && m_filterOnRun == m_run));

  if (!shouldWrite)
    return;

  std::vector<MonitorElement*> items(dstore->getAllContents("", m_run, 0));
  for (std::vector<MonitorElement*>::iterator it = items.begin(), itEnd = items.end(); it != itEnd; ++it) {
    assert((*it)->getScope() == MonitorElementData::Scope::RUN);
    std::map<unsigned int, unsigned int>::iterator itFound = m_dqmKindToTypeIndex.find((int)(*it)->kind());
    assert(itFound != m_dqmKindToTypeIndex.end());
    m_treeHelpers[itFound->second]->fill(*it);
  }

  const edm::ProcessHistoryID& id = iRun.processHistoryID();
  std::vector<edm::ProcessHistoryID>::iterator itFind = std::find(m_seenHistories.begin(), m_seenHistories.end(), id);
  if (itFind == m_seenHistories.end()) {
    m_processHistoryRegistry.registerProcessHistory(iRun.processHistory());
    m_presentHistoryIndex = m_seenHistories.size();
    m_seenHistories.push_back(id);
  } else {
    m_presentHistoryIndex = itFind - m_seenHistories.begin();
  }

  //Now store the relationship between run/lumi and indices in the other TTrees
  unsigned int typeIndex = 0;
  for (std::vector<std::shared_ptr<TreeHelperBase> >::iterator it = m_treeHelpers.begin(), itEnd = m_treeHelpers.end();
       it != itEnd;
       ++it, ++typeIndex) {
    if ((*it)->wasFilled()) {
      m_type = typeIndex;
      (*it)->getRangeAndReset(m_firstIndex, m_lastIndex);
      tbb::this_task_arena::isolate([&] { m_indicesTree->Fill(); });
    }
  }

  edm::Service<edm::JobReport> jr;
  jr->reportRunNumber(m_jrToken, m_run);
}

void DQMRootOutputModule::reallyCloseFile() {
  startEndFile();
  finishEndFile();
}

void DQMRootOutputModule::startEndFile() {
  //std::cout << "DQMRootOutputModule::startEndFile"<< std::endl;
  //fill in the meta data
  m_file->cd();
  TDirectory* metaDataDirectory = m_file->mkdir(kMetaDataDirectory);

  //Write out the Process History
  TTree* processHistoryTree = new TTree(kProcessHistoryTree, kProcessHistoryTree);
  processHistoryTree->SetDirectory(metaDataDirectory);

  unsigned int index = 0;
  processHistoryTree->Branch(kPHIndexBranch, &index);
  std::string processName;
  processHistoryTree->Branch(kProcessConfigurationProcessNameBranch, &processName);
  std::string parameterSetID;
  processHistoryTree->Branch(kProcessConfigurationParameterSetIDBranch, &parameterSetID);
  std::string releaseVersion;
  processHistoryTree->Branch(kProcessConfigurationReleaseVersion, &releaseVersion);
  std::string passID;
  processHistoryTree->Branch(kProcessConfigurationPassID, &passID);

  for (std::vector<edm::ProcessHistoryID>::iterator it = m_seenHistories.begin(), itEnd = m_seenHistories.end();
       it != itEnd;
       ++it) {
    const edm::ProcessHistory* history = m_processHistoryRegistry.getMapped(*it);
    assert(nullptr != history);
    index = 0;
    for (edm::ProcessHistory::collection_type::const_iterator itPC = history->begin(), itPCEnd = history->end();
         itPC != itPCEnd;
         ++itPC, ++index) {
      processName = itPC->processName();
      releaseVersion = itPC->releaseVersion();
      passID = itPC->passID();
      parameterSetID = itPC->parameterSetID().compactForm();
      tbb::this_task_arena::isolate([&] { processHistoryTree->Fill(); });
    }
  }

  //Store the ParameterSets
  TTree* parameterSetsTree = new TTree(kParameterSetTree, kParameterSetTree);
  parameterSetsTree->SetDirectory(metaDataDirectory);
  std::string blob;
  parameterSetsTree->Branch(kParameterSetBranch, &blob);

  edm::pset::Registry* psr = edm::pset::Registry::instance();
  assert(nullptr != psr);
  for (edm::pset::Registry::const_iterator it = psr->begin(), itEnd = psr->end(); it != itEnd; ++it) {
    blob.clear();
    it->second.toString(blob);
    tbb::this_task_arena::isolate([&] { parameterSetsTree->Fill(); });
  }
}

void DQMRootOutputModule::finishEndFile() {
  //std::cout << "DQMRootOutputModule::finishEndFile"<< std::endl;
  m_file->Write();
  m_file->Close();
  edm::Service<edm::JobReport> jr;
  jr->outputFileClosed(m_jrToken);
}

//
// const member functions
//

//
// static member functions
//
void DQMRootOutputModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.addUntracked<std::string>("fileName");
  desc.addUntracked<std::string>("logicalFileName", "");
  desc.addUntracked<unsigned int>("filterOnRun", 0)
      ->setComment("Only write the run with this run number. 0 means write all runs.");
  desc.addOptionalUntracked<int>("splitLevel", 99)
      ->setComment("UNUSED Only here to allow older configurations written for PoolOutputModule to work.");
  const std::vector<std::string> keep = {"drop *", "keep DQMToken_*_*_*"};
  edm::one::OutputModule<>::fillDescription(desc, keep);

  edm::ParameterSetDescription dataSet;
  dataSet.setAllowAnything();
  desc.addUntracked<edm::ParameterSetDescription>("dataset", dataSet)
      ->setComment("PSet is only used by Data Operations and not by this module.");

  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(DQMRootOutputModule);
