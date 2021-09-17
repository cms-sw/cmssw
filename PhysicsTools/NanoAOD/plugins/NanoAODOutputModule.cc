// -*- C++ -*-
//
// Package:     PhysicsTools/NanoAODOutput
// Class  :     NanoAODOutputModule
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Mon, 07 Aug 2017 14:21:41 GMT
//

// system include files
#include <algorithm>
#include <memory>

#include "Compression.h"
#include "TFile.h"
#include "TObjString.h"
#include "TROOT.h"
#include "TTree.h"
#include <string>

// user include files
#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Utilities/interface/Digest.h"
#include "IOPool/Provenance/interface/CommonProvenanceFiller.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/NanoAOD/interface/UniqueString.h"
#include "PhysicsTools/NanoAOD/plugins/TableOutputBranches.h"
#include "PhysicsTools/NanoAOD/plugins/TriggerOutputBranches.h"
#include "PhysicsTools/NanoAOD/plugins/EventStringOutputBranches.h"
#include "PhysicsTools/NanoAOD/plugins/SummaryTableOutputBranches.h"

#include <iostream>

#include "tbb/task_arena.h"

class NanoAODOutputModule : public edm::one::OutputModule<> {
public:
  NanoAODOutputModule(edm::ParameterSet const& pset);
  ~NanoAODOutputModule() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void write(edm::EventForOutput const& e) override;
  void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override;
  void writeRun(edm::RunForOutput const&) override;
  bool isFileOpen() const override;
  void openFile(edm::FileBlock const&) override;
  void reallyCloseFile() override;

  std::string m_fileName;
  std::string m_logicalFileName;
  int m_compressionLevel;
  int m_eventsSinceFlush{0};
  std::string m_compressionAlgorithm;
  bool m_writeProvenance;
  bool m_fakeName;  //crab workaround, remove after crab is fixed
  int m_autoFlush;
  edm::ProcessHistoryRegistry m_processHistoryRegistry;
  edm::JobReport::Token m_jrToken;
  std::unique_ptr<TFile> m_file;
  std::unique_ptr<TTree> m_tree, m_lumiTree, m_runTree, m_metaDataTree, m_parameterSetsTree;

  static constexpr int m_firstFlush{1000};

  class CommonEventBranches {
  public:
    void branch(TTree& tree) {
      tree.Branch("run", &m_run, "run/i");
      tree.Branch("luminosityBlock", &m_luminosityBlock, "luminosityBlock/i");
      tree.Branch("event", &m_event, "event/l");
    }
    void fill(const edm::EventID& id) {
      m_run = id.run();
      m_luminosityBlock = id.luminosityBlock();
      m_event = id.event();
    }

  private:
    UInt_t m_run;
    UInt_t m_luminosityBlock;
    ULong64_t m_event;
  } m_commonBranches;

  class CommonLumiBranches {
  public:
    void branch(TTree& tree) {
      tree.Branch("run", &m_run, "run/i");
      tree.Branch("luminosityBlock", &m_luminosityBlock, "luminosityBlock/i");
    }
    void fill(const edm::LuminosityBlockID& id) {
      m_run = id.run();
      m_luminosityBlock = id.value();
    }

  private:
    UInt_t m_run;
    UInt_t m_luminosityBlock;
  } m_commonLumiBranches;

  class CommonRunBranches {
  public:
    void branch(TTree& tree) { tree.Branch("run", &m_run, "run/i"); }
    void fill(const edm::RunID& id) { m_run = id.run(); }

  private:
    UInt_t m_run;
  } m_commonRunBranches;

  std::vector<TableOutputBranches> m_tables;
  std::vector<TriggerOutputBranches> m_triggers;
  bool m_triggers_areSorted = false;
  std::vector<EventStringOutputBranches> m_evstrings;

  std::vector<SummaryTableOutputBranches> m_runTables;
  std::vector<SummaryTableOutputBranches> m_lumiTables;
  std::vector<TableOutputBranches> m_runFlatTables;

  std::vector<std::pair<std::string, edm::EDGetToken>> m_nanoMetadata;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
NanoAODOutputModule::NanoAODOutputModule(edm::ParameterSet const& pset)
    : edm::one::OutputModuleBase::OutputModuleBase(pset),
      edm::one::OutputModule<>(pset),
      m_fileName(pset.getUntrackedParameter<std::string>("fileName")),
      m_logicalFileName(pset.getUntrackedParameter<std::string>("logicalFileName")),
      m_compressionLevel(pset.getUntrackedParameter<int>("compressionLevel")),
      m_compressionAlgorithm(pset.getUntrackedParameter<std::string>("compressionAlgorithm")),
      m_writeProvenance(pset.getUntrackedParameter<bool>("saveProvenance", true)),
      m_fakeName(pset.getUntrackedParameter<bool>("fakeNameForCrab", false)),
      m_autoFlush(pset.getUntrackedParameter<int>("autoFlush", -10000000)),
      m_processHistoryRegistry() {}

NanoAODOutputModule::~NanoAODOutputModule() {}

void NanoAODOutputModule::write(edm::EventForOutput const& iEvent) {
  //Get data from 'e' and write it to the file
  edm::Service<edm::JobReport> jr;
  jr->eventWrittenToFile(m_jrToken, iEvent.id().run(), iEvent.id().event());

  if (m_autoFlush) {
    int64_t events = m_tree->GetEntriesFast();
    if (events == m_firstFlush) {
      m_tree->FlushBaskets();
      float maxMemory;
      if (m_autoFlush > 0) {
        // Estimate the memory we'll be using at the first full flush by
        // linearly scaling the number of events.
        float percentClusterDone = m_firstFlush / static_cast<float>(m_autoFlush);
        maxMemory = static_cast<float>(m_tree->GetTotBytes()) / percentClusterDone;
      } else if (m_tree->GetZipBytes() == 0) {
        maxMemory = 100 * 1024 * 1024;  // Degenerate case of no information in the tree; arbitrary value
      } else {
        // Estimate the memory we'll be using by scaling the current compression ratio.
        float cxnRatio = m_tree->GetTotBytes() / static_cast<float>(m_tree->GetZipBytes());
        maxMemory = -m_autoFlush * cxnRatio;
        float percentBytesDone = -m_tree->GetZipBytes() / static_cast<float>(m_autoFlush);
        m_autoFlush = m_firstFlush / percentBytesDone;
      }
      //std::cout << "OptimizeBaskets: total bytes " << m_tree->GetTotBytes() << std::endl;
      //std::cout << "OptimizeBaskets: zip bytes " << m_tree->GetZipBytes() << std::endl;
      //std::cout << "OptimizeBaskets: autoFlush " << m_autoFlush << std::endl;
      //std::cout << "OptimizeBaskets: maxMemory " << static_cast<uint32_t>(maxMemory) << std::endl;
      //m_tree->OptimizeBaskets(static_cast<uint32_t>(maxMemory), 1, "d");
      m_tree->OptimizeBaskets(static_cast<uint32_t>(maxMemory), 1, "");
    }
    if (m_eventsSinceFlush == m_autoFlush) {
      m_tree->FlushBaskets();
      m_eventsSinceFlush = 0;
    }
    m_eventsSinceFlush++;
  }

  m_commonBranches.fill(iEvent.id());
  // fill all tables, starting from main tables and then doing extension tables
  for (unsigned int extensions = 0; extensions <= 1; ++extensions) {
    for (auto& t : m_tables)
      t.fill(iEvent, *m_tree, extensions);
  }
  if (!m_triggers_areSorted) {  // sort triggers/flags in inverse processHistory order, to save without any special label the most recent ones
    std::vector<std::string> pnames;
    for (auto& p : iEvent.processHistory())
      pnames.push_back(p.processName());
    std::sort(m_triggers.begin(), m_triggers.end(), [pnames](TriggerOutputBranches& a, TriggerOutputBranches& b) {
      return ((std::find(pnames.begin(), pnames.end(), a.processName()) - pnames.begin()) >
              (std::find(pnames.begin(), pnames.end(), b.processName()) - pnames.begin()));
    });
    m_triggers_areSorted = true;
  }
  // fill triggers
  for (auto& t : m_triggers)
    t.fill(iEvent, *m_tree);
  // fill event branches
  for (auto& t : m_evstrings)
    t.fill(iEvent, *m_tree);
  tbb::this_task_arena::isolate([&] { m_tree->Fill(); });

  m_processHistoryRegistry.registerProcessHistory(iEvent.processHistory());
}

void NanoAODOutputModule::writeLuminosityBlock(edm::LuminosityBlockForOutput const& iLumi) {
  edm::Service<edm::JobReport> jr;
  jr->reportLumiSection(m_jrToken, iLumi.id().run(), iLumi.id().value());

  m_commonLumiBranches.fill(iLumi.id());

  for (auto& t : m_lumiTables)
    t.fill(iLumi, *m_lumiTree);

  tbb::this_task_arena::isolate([&] { m_lumiTree->Fill(); });

  m_processHistoryRegistry.registerProcessHistory(iLumi.processHistory());
}

void NanoAODOutputModule::writeRun(edm::RunForOutput const& iRun) {
  edm::Service<edm::JobReport> jr;
  jr->reportRunNumber(m_jrToken, iRun.id().run());

  m_commonRunBranches.fill(iRun.id());

  for (auto& t : m_runTables)
    t.fill(iRun, *m_runTree);

  for (unsigned int extensions = 0; extensions <= 1; ++extensions) {
    for (auto& t : m_runFlatTables)
      t.fill(iRun, *m_runTree, extensions);
  }

  edm::Handle<nanoaod::UniqueString> hstring;
  for (const auto& p : m_nanoMetadata) {
    iRun.getByToken(p.second, hstring);
    TObjString* tos = dynamic_cast<TObjString*>(m_file->Get(p.first.c_str()));
    if (tos) {
      if (hstring->str() != tos->GetString())
        throw cms::Exception("LogicError", "Inconsistent nanoMetadata " + p.first + " (" + hstring->str() + ")");
    } else {
      auto ostr = std::make_unique<TObjString>(hstring->str().c_str());
      m_file->WriteTObject(ostr.release(), p.first.c_str());
    }
  }

  tbb::this_task_arena::isolate([&] { m_runTree->Fill(); });

  m_processHistoryRegistry.registerProcessHistory(iRun.processHistory());
}

bool NanoAODOutputModule::isFileOpen() const { return nullptr != m_file.get(); }

void NanoAODOutputModule::openFile(edm::FileBlock const&) {
  m_file = std::make_unique<TFile>(m_fileName.c_str(), "RECREATE", "", m_compressionLevel);
  edm::Service<edm::JobReport> jr;
  cms::Digest branchHash;
  m_jrToken = jr->outputFileOpened(m_fileName,
                                   m_logicalFileName,
                                   std::string(),
                                   m_fakeName ? "PoolOutputModule" : "NanoAODOutputModule",
                                   description().moduleLabel(),
                                   edm::createGlobalIdentifier(),
                                   std::string(),
                                   branchHash.digest().toString(),
                                   std::vector<std::string>());

  if (m_compressionAlgorithm == std::string("ZLIB")) {
    m_file->SetCompressionAlgorithm(ROOT::kZLIB);
  } else if (m_compressionAlgorithm == std::string("LZMA")) {
    m_file->SetCompressionAlgorithm(ROOT::kLZMA);
  } else {
    throw cms::Exception("Configuration")
        << "NanoAODOutputModule configured with unknown compression algorithm '" << m_compressionAlgorithm << "'\n"
        << "Allowed compression algorithms are ZLIB and LZMA\n";
  }
  /* Setup file structure here */
  m_tables.clear();
  m_triggers.clear();
  m_triggers_areSorted = false;
  m_evstrings.clear();
  m_runTables.clear();
  m_lumiTables.clear();
  m_runFlatTables.clear();
  const auto& keeps = keptProducts();
  for (const auto& keep : keeps[edm::InEvent]) {
    if (keep.first->className() == "nanoaod::FlatTable")
      m_tables.emplace_back(keep.first, keep.second);
    else if (keep.first->className() == "edm::TriggerResults") {
      m_triggers.emplace_back(keep.first, keep.second);
    } else if (keep.first->className() == "std::basic_string<char,std::char_traits<char> >" &&
               keep.first->productInstanceName() == "genModel") {  // friendlyClassName == "String"
      m_evstrings.emplace_back(keep.first, keep.second, true);     // update only at lumiBlock transitions
    } else
      throw cms::Exception("Configuration", "NanoAODOutputModule cannot handle class " + keep.first->className());
  }

  for (const auto& keep : keeps[edm::InLumi]) {
    if (keep.first->className() == "nanoaod::MergeableCounterTable")
      m_lumiTables.push_back(SummaryTableOutputBranches(keep.first, keep.second));
    else if (keep.first->className() == "nanoaod::UniqueString" && keep.first->moduleLabel() == "nanoMetadata")
      m_nanoMetadata.emplace_back(keep.first->productInstanceName(), keep.second);
    else
      throw cms::Exception(
          "Configuration",
          "NanoAODOutputModule cannot handle class " + keep.first->className() + " in LuminosityBlock branch");
  }

  for (const auto& keep : keeps[edm::InRun]) {
    if (keep.first->className() == "nanoaod::MergeableCounterTable")
      m_runTables.push_back(SummaryTableOutputBranches(keep.first, keep.second));
    else if (keep.first->className() == "nanoaod::UniqueString" && keep.first->moduleLabel() == "nanoMetadata")
      m_nanoMetadata.emplace_back(keep.first->productInstanceName(), keep.second);
    else if (keep.first->className() == "nanoaod::FlatTable")
      m_runFlatTables.emplace_back(keep.first, keep.second);
    else
      throw cms::Exception("Configuration",
                           "NanoAODOutputModule cannot handle class " + keep.first->className() + " in Run branch");
  }

  // create the trees
  m_tree = std::make_unique<TTree>("Events", "Events");
  m_tree->SetAutoSave(0);
  m_tree->SetAutoFlush(0);
  m_commonBranches.branch(*m_tree);

  m_lumiTree = std::make_unique<TTree>("LuminosityBlocks", "LuminosityBlocks");
  m_lumiTree->SetAutoSave(0);
  m_commonLumiBranches.branch(*m_lumiTree);

  m_runTree = std::make_unique<TTree>("Runs", "Runs");
  m_runTree->SetAutoSave(0);
  m_commonRunBranches.branch(*m_runTree);

  if (m_writeProvenance) {
    m_metaDataTree = std::make_unique<TTree>(edm::poolNames::metaDataTreeName().c_str(), "Job metadata");
    m_metaDataTree->SetAutoSave(0);
    m_parameterSetsTree = std::make_unique<TTree>(edm::poolNames::parameterSetsTreeName().c_str(), "Parameter sets");
    m_parameterSetsTree->SetAutoSave(0);
  }
}
void NanoAODOutputModule::reallyCloseFile() {
  if (m_writeProvenance) {
    int basketSize = 16384;  // fixme configurable?
    edm::fillParameterSetBranch(m_parameterSetsTree.get(), basketSize);
    edm::fillProcessHistoryBranch(m_metaDataTree.get(), basketSize, m_processHistoryRegistry);
    if (m_metaDataTree->GetNbranches() != 0) {
      m_metaDataTree->SetEntries(-1);
    }
    if (m_parameterSetsTree->GetNbranches() != 0) {
      m_parameterSetsTree->SetEntries(-1);
    }
  }
  m_file->Write();
  m_file->Close();
  m_file.reset();
  m_tree.release();               // apparently root has ownership
  m_lumiTree.release();           //
  m_runTree.release();            //
  m_metaDataTree.release();       //
  m_parameterSetsTree.release();  //
  edm::Service<edm::JobReport> jr;
  jr->outputFileClosed(m_jrToken);
}

void NanoAODOutputModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.addUntracked<std::string>("fileName");
  desc.addUntracked<std::string>("logicalFileName", "");

  desc.addUntracked<int>("compressionLevel", 9)->setComment("ROOT compression level of output file.");
  desc.addUntracked<std::string>("compressionAlgorithm", "ZLIB")
      ->setComment("Algorithm used to compress data in the ROOT output file, allowed values are ZLIB and LZMA");
  desc.addUntracked<bool>("saveProvenance", true)
      ->setComment("Save process provenance information, e.g. for edmProvDump");
  desc.addUntracked<bool>("fakeNameForCrab", false)
      ->setComment(
          "Change the OutputModule name in the fwk job report to fake PoolOutputModule. This is needed to run on cran "
          "(and publish) till crab is fixed");
  desc.addUntracked<int>("autoFlush", -10000000)->setComment("Autoflush parameter for ROOT file");

  //replace with whatever you want to get from the EDM by default
  const std::vector<std::string> keep = {"drop *",
                                         "keep nanoaodFlatTable_*Table_*_*",
                                         "keep edmTriggerResults_*_*_*",
                                         "keep String_*_genModel_*",
                                         "keep nanoaodMergeableCounterTable_*Table_*_*",
                                         "keep nanoaodUniqueString_nanoMetadata_*_*"};
  edm::one::OutputModule<>::fillDescription(desc, keep);

  //Used by Workflow management for their own meta data
  edm::ParameterSetDescription dataSet;
  dataSet.setAllowAnything();
  desc.addUntracked<edm::ParameterSetDescription>("dataset", dataSet)
      ->setComment("PSet is only used by Data Operations and not by this module.");

  edm::ParameterSetDescription branchSet;
  branchSet.setAllowAnything();
  desc.add<edm::ParameterSetDescription>("branches", branchSet);

  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(NanoAODOutputModule);
