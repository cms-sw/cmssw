// -*- C++ -*-
//
// Package:     L1TriggerScouting/Utilities
// Class  :     OrbitNanoAODOutputModule
//
// Implementation:
//     Adapt from NanoAODOutputModule for OrbitFlatTable
//     This handles rotating OrbitCollection to Event
//
//
// Author original version: Giovanni Petrucciani
//        adapted by Patin Inkaew

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
#include "L1TriggerScouting/Utilities/plugins/OrbitTableOutputBranches.h"
#include "L1TriggerScouting/Utilities/plugins/SelectedBxTableOutputBranches.h"

#include "oneapi/tbb/task_arena.h"

class OrbitNanoAODOutputModule : public edm::one::OutputModule<> {
public:
  OrbitNanoAODOutputModule(edm::ParameterSet const& pset);
  ~OrbitNanoAODOutputModule() override;

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
  bool m_skipEmptyBXs;
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
      tree.Branch("bunchCrossing", &m_bunchCrossing, "bunchCrossing/i");
      tree.Branch("orbitNumber", &m_orbitNumber, "orbitNumber/i");
    }
    void fill(const edm::EventAuxiliary& aux) {
      m_run = aux.id().run();
      m_luminosityBlock = aux.id().luminosityBlock();
      m_orbitNumber = aux.id().event();  // in L1Scouting, one processing event is one orbit
    }
    void setBx(unsigned bx) { m_bunchCrossing = bx; }

  private:
    UInt_t m_run;
    UInt_t m_luminosityBlock;
    UInt_t m_bunchCrossing;
    UInt_t m_orbitNumber;
  } m_commonBranches;

  class CommonLumiBranches {
  public:
    void branch(TTree& tree) {
      tree.Branch("run", &m_run, "run/i");
      tree.Branch("luminosityBlock", &m_luminosityBlock, "luminosityBlock/i");
      tree.Branch("nOrbits", &m_orbits, "nOrbits/i");
    }
    void fill(const edm::LuminosityBlockID& id, unsigned int nOrbits) {
      m_run = id.run();
      m_luminosityBlock = id.value();
      m_orbits = nOrbits;
    }

  private:
    UInt_t m_run;
    UInt_t m_luminosityBlock;
    UInt_t m_orbits;
  } m_commonLumiBranches;

  class CommonRunBranches {
  public:
    void branch(TTree& tree) { tree.Branch("run", &m_run, "run/i"); }
    void fill(const edm::RunID& id) { m_run = id.run(); }

  private:
    UInt_t m_run;
  } m_commonRunBranches;

  std::vector<OrbitTableOutputBranches> m_tables;
  std::vector<SelectedBxTableOutputBranches> m_selbxs;
  unsigned int m_nOrbits;

  std::vector<std::pair<std::string, edm::EDGetToken>> m_nanoMetadata;

  edm::EDGetTokenT<std::vector<unsigned>> m_bxMaskToken;
  std::vector<unsigned> allBXs_;
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
OrbitNanoAODOutputModule::OrbitNanoAODOutputModule(edm::ParameterSet const& pset)
    : edm::one::OutputModuleBase::OutputModuleBase(pset),
      edm::one::OutputModule<>(pset),
      m_fileName(pset.getUntrackedParameter<std::string>("fileName")),
      m_logicalFileName(pset.getUntrackedParameter<std::string>("logicalFileName")),
      m_compressionLevel(pset.getUntrackedParameter<int>("compressionLevel")),
      m_compressionAlgorithm(pset.getUntrackedParameter<std::string>("compressionAlgorithm")),
      m_skipEmptyBXs(pset.getParameter<bool>("skipEmptyBXs")),
      m_writeProvenance(pset.getUntrackedParameter<bool>("saveProvenance", true)),
      m_fakeName(pset.getUntrackedParameter<bool>("fakeNameForCrab", false)),
      m_autoFlush(pset.getUntrackedParameter<int>("autoFlush", -10000000)),
      m_processHistoryRegistry(),
      m_nOrbits(0) {
  edm::InputTag bxMask = pset.getParameter<edm::InputTag>("selectedBx");
  if (!bxMask.label().empty()) {
    m_bxMaskToken = consumes<std::vector<unsigned>>(bxMask);
  } else {
    allBXs_.resize(l1ScoutingRun3::OrbitFlatTable::NBX);
    for (unsigned int i = 0; i < l1ScoutingRun3::OrbitFlatTable::NBX; ++i) {
      allBXs_[i] = i + 1;
    }
  }
}

OrbitNanoAODOutputModule::~OrbitNanoAODOutputModule() {}

void OrbitNanoAODOutputModule::write(edm::EventForOutput const& iEvent) {
  //Get data from 'e' and write it to the file
  edm::Service<edm::JobReport> jr;
  jr->eventWrittenToFile(m_jrToken, iEvent.id().run(), iEvent.id().event());
  m_nOrbits++;

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

  m_commonBranches.fill(iEvent.eventAuxiliary());
  // fill all tables, starting from main tables and then doing extension tables
  for (unsigned int extensions = 0; extensions <= 1; ++extensions) {
    for (auto& t : m_tables) {
      t.beginFill(iEvent, *m_tree, extensions);
    }
  }
  // get m_table, read parameters, book branches, etc.
  for (auto& t : m_selbxs) {
    t.beginFill(iEvent, *m_tree);
  }
  // get a lust of selected BXs to be filled
  const std::vector<unsigned>* selbx = &allBXs_;
  if (!m_bxMaskToken.isUninitialized()) {
    edm::Handle<std::vector<unsigned>> handle;
    iEvent.getByToken(m_bxMaskToken, handle);
    selbx = &*handle;
  }
  // convert from orbit as event to collision as event
  tbb::this_task_arena::isolate([&] {
    for (unsigned bx : *selbx) {
      if (m_skipEmptyBXs) {
        bool empty = true;
        for (auto& t : m_tables) {
          if (t.hasBx(bx)) {
            empty = false;
            break;
          }
        }
        if (empty) {
          continue;
        }
      }

      m_commonBranches.setBx(bx);
      for (auto& t : m_tables) {
        t.fillBx(bx);
      }
      for (auto& t : m_selbxs) {
        t.fillBx(bx);
      }
      m_tree->Fill();
    }  // bx loop
  });

  // set m_table to nullptr
  for (auto& t : m_tables) {
    t.endFill();
  }
  for (auto& t : m_selbxs) {
    t.endFill();
  }
  m_processHistoryRegistry.registerProcessHistory(iEvent.processHistory());
}

void OrbitNanoAODOutputModule::writeLuminosityBlock(edm::LuminosityBlockForOutput const& iLumi) {
  edm::Service<edm::JobReport> jr;
  jr->reportLumiSection(m_jrToken, iLumi.id().run(), iLumi.id().value());

  m_commonLumiBranches.fill(iLumi.id(), m_nOrbits);

  tbb::this_task_arena::isolate([&] { m_lumiTree->Fill(); });

  m_processHistoryRegistry.registerProcessHistory(iLumi.processHistory());

  m_nOrbits = 0;
}

void OrbitNanoAODOutputModule::writeRun(edm::RunForOutput const& iRun) {
  edm::Service<edm::JobReport> jr;
  jr->reportRunNumber(m_jrToken, iRun.id().run());

  m_commonRunBranches.fill(iRun.id());

  tbb::this_task_arena::isolate([&] { m_runTree->Fill(); });

  m_processHistoryRegistry.registerProcessHistory(iRun.processHistory());
}

bool OrbitNanoAODOutputModule::isFileOpen() const { return nullptr != m_file.get(); }

void OrbitNanoAODOutputModule::openFile(edm::FileBlock const&) {
  m_file = std::make_unique<TFile>(m_fileName.c_str(), "RECREATE", "", m_compressionLevel);
  edm::Service<edm::JobReport> jr;
  cms::Digest branchHash;
  m_jrToken = jr->outputFileOpened(m_fileName,
                                   m_logicalFileName,
                                   std::string(),
                                   m_fakeName ? "PoolOutputModule" : "OrbitNanoAODOutputModule",
                                   description().moduleLabel(),
                                   edm::createGlobalIdentifier(),
                                   std::string(),
                                   branchHash.digest().toString(),
                                   std::vector<std::string>());

  if (m_compressionAlgorithm == std::string("ZLIB")) {
    m_file->SetCompressionAlgorithm(ROOT::RCompressionSetting::EAlgorithm::kZLIB);
  } else if (m_compressionAlgorithm == std::string("LZMA")) {
    m_file->SetCompressionAlgorithm(ROOT::RCompressionSetting::EAlgorithm::kLZMA);
  } else if (m_compressionAlgorithm == std::string("ZSTD")) {
    m_file->SetCompressionAlgorithm(ROOT::RCompressionSetting::EAlgorithm::kZSTD);
  } else if (m_compressionAlgorithm == std::string("LZ4")) {
    m_file->SetCompressionAlgorithm(ROOT::RCompressionSetting::EAlgorithm::kLZ4);
  } else {
    throw cms::Exception("Configuration")
        << "OrbitNanoAODOutputModule configured with unknown compression algorithm '" << m_compressionAlgorithm << "'\n"
        << "Allowed compression algorithms are ZLIB, LZMA, ZSTD, and LZ4\n";
  }
  /* Setup file structure here */
  m_tables.clear();
  const auto& keeps = keptProducts();
  for (const auto& keep : keeps[edm::InEvent]) {
    if (keep.first->className() == "l1ScoutingRun3::OrbitFlatTable")
      m_tables.emplace_back(keep.first, keep.second);
    else if (keep.first->className() == "std::vector<unsigned int>")
      m_selbxs.emplace_back(keep.first, keep.second);
    else
      throw cms::Exception("Configuration", "OrbitNanoAODOutputModule cannot handle class " + keep.first->className());
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
void OrbitNanoAODOutputModule::reallyCloseFile() {
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

void OrbitNanoAODOutputModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.addUntracked<std::string>("fileName");
  desc.addUntracked<std::string>("logicalFileName", "");

  desc.addUntracked<int>("compressionLevel", 9)->setComment("ROOT compression level of output file.");
  desc.addUntracked<std::string>("compressionAlgorithm", "ZLIB")
      ->setComment("Algorithm used to compress data in the ROOT output file, allowed values are ZLIB and LZMA");
  desc.add<bool>("skipEmptyBXs", false)->setComment("Skip BXs where all input collections are empty");
  desc.addUntracked<bool>("saveProvenance", true)
      ->setComment("Save process provenance information, e.g. for edmProvDump");
  desc.addUntracked<bool>("fakeNameForCrab", false)
      ->setComment(
          "Change the OutputModule name in the fwk job report to fake PoolOutputModule. This is needed to run on "
          "crab "
          "(and publish) till crab is fixed");
  desc.addUntracked<int>("autoFlush", -10000000)->setComment("Autoflush parameter for ROOT file");

  //replace with whatever you want to get from the EDM by default
  const std::vector<std::string> keep = {"drop *", "keep l1ScoutingRun3OrbitFlatTable_*Table_*_*"};
  edm::one::OutputModule<>::fillDescription(desc, keep);

  //Used by Workflow management for their own meta data
  edm::ParameterSetDescription dataSet;
  dataSet.setAllowAnything();
  desc.addUntracked<edm::ParameterSetDescription>("dataset", dataSet)
      ->setComment("PSet is only used by Data Operations and not by this module.");

  edm::ParameterSetDescription branchSet;
  branchSet.setAllowAnything();
  desc.add<edm::ParameterSetDescription>("branches", branchSet);

  desc.add<edm::InputTag>("selectedBx", edm::InputTag())->setComment("selected Bx (1-3564)");
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(OrbitNanoAODOutputModule);
