#include "Compression.h"
#include "TObjString.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Provenance/interface/CommonProvenanceFiller.h"
#include "PhysicsTools/NanoAOD/interface/NanoAODOutputModuleBase.h"

NanoAODOutputModuleBase::NanoAODOutputModuleBase(edm::ParameterSet const& pset)
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

void NanoAODOutputModuleBase::write(edm::EventForOutput const& iEvent) {
  //Get data from 'e' and write it to the file
  edm::Service<edm::JobReport> jr;
  jr->eventWrittenToFile(m_jrToken, iEvent.id().run(), iEvent.id().event());

  if (m_autoFlush) {
    int64_t events = m_eventTree->GetEntriesFast();
    if (events == kFirstFlush) {
      m_eventTree->FlushBaskets();
      float maxMemory;
      if (m_autoFlush > 0) {
        // Estimate the memory we'll be using at the first full flush by
        // linearly scaling the number of events.
        float percentClusterDone = kFirstFlush / static_cast<float>(m_autoFlush);
        maxMemory = static_cast<float>(m_eventTree->GetTotBytes()) / percentClusterDone;
      } else if (m_eventTree->GetZipBytes() == 0) {
        maxMemory = 100 * 1024 * 1024;  // Degenerate case of no information in the tree; arbitrary value
      } else {
        // Estimate the memory we'll be using by scaling the current compression ratio.
        float cxnRatio = m_eventTree->GetTotBytes() / static_cast<float>(m_eventTree->GetZipBytes());
        maxMemory = -m_autoFlush * cxnRatio;
        float percentBytesDone = -m_eventTree->GetZipBytes() / static_cast<float>(m_autoFlush);
        m_autoFlush = kFirstFlush / percentBytesDone;
      }
      m_eventTree->OptimizeBaskets(static_cast<uint32_t>(maxMemory), 1, "");
    }
    if (m_eventsSinceFlush == m_autoFlush) {
      m_eventTree->FlushBaskets();
      m_eventsSinceFlush = 0;
    }
    m_eventsSinceFlush++;
  }

  m_processHistoryRegistry.registerProcessHistory(iEvent.processHistory());

  writeEventTree(iEvent);
}

void NanoAODOutputModuleBase::writeLuminosityBlock(edm::LuminosityBlockForOutput const& iLumi) {
  edm::Service<edm::JobReport> jr;
  jr->reportLumiSection(m_jrToken, iLumi.id().run(), iLumi.id().value());

  m_processHistoryRegistry.registerProcessHistory(iLumi.processHistory());

  writeLuminosityBlockTree(iLumi);
}

void NanoAODOutputModuleBase::writeRun(edm::RunForOutput const& iRun) {
  edm::Service<edm::JobReport> jr;
  jr->reportRunNumber(m_jrToken, iRun.id().run());

  m_processHistoryRegistry.registerProcessHistory(iRun.processHistory());

  writeRunTree(iRun);
}

bool NanoAODOutputModuleBase::isFileOpen() const { return nullptr != m_file.get(); }

void NanoAODOutputModuleBase::openFile(edm::FileBlock const&) {
  m_file = std::make_unique<TFile>(m_fileName.c_str(), "RECREATE", "", m_compressionLevel);
  edm::Service<edm::JobReport> jr;
  cms::Digest branchHash;
  m_jrToken = jr->outputFileOpened(m_fileName,
                                   m_logicalFileName,
                                   std::string(),
                                   m_fakeName ? kFakeOutputModuleType : description().moduleName(),
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
        << "NanoAOD OutputModule configured with unknown compression algorithm '" << m_compressionAlgorithm << "'\n"
        << "Allowed compression algorithms are ZLIB, LZMA, ZSTD, and LZ4\n";
  }

  initTables();

  // create the trees
  m_eventTree = std::make_unique<TTree>("Events", "Events");
  m_eventTree->SetAutoSave(0);
  m_eventTree->SetAutoFlush(0);
  initEventTree(*m_eventTree);

  m_lumiTree = std::make_unique<TTree>("LuminosityBlocks", "LuminosityBlocks");
  m_lumiTree->SetAutoSave(0);
  initLuminosityBlockTree(*m_lumiTree);

  m_runTree = std::make_unique<TTree>("Runs", "Runs");
  m_runTree->SetAutoSave(0);
  initRunTree(*m_runTree);

  if (m_writeProvenance) {
    m_metaDataTree = std::make_unique<TTree>(edm::poolNames::metaDataTreeName().c_str(), "Job metadata");
    m_metaDataTree->SetAutoSave(0);
    m_parameterSetsTree = std::make_unique<TTree>(edm::poolNames::parameterSetsTreeName().c_str(), "Parameter sets");
    m_parameterSetsTree->SetAutoSave(0);
  }
}
void NanoAODOutputModuleBase::reallyCloseFile() {
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
  m_eventTree.release();          // apparently ROOT has ownership
  m_lumiTree.release();           //
  m_runTree.release();            //
  m_metaDataTree.release();       //
  m_parameterSetsTree.release();  //
  edm::Service<edm::JobReport> jr;
  jr->outputFileClosed(m_jrToken);
}

void NanoAODOutputModuleBase::fillDescription(edm::ParameterSetDescription& desc) {
  desc.addUntracked<std::string>("fileName");
  desc.addUntracked<std::string>("logicalFileName", "");

  desc.addUntracked<int>("compressionLevel", 9)->setComment("ROOT compression level of output file.");
  desc.addUntracked<std::string>("compressionAlgorithm", "ZLIB")
      ->setComment("Algorithm used to compress data in the ROOT output file, allowed values are ZLIB and LZMA");
  desc.addUntracked<bool>("saveProvenance", true)
      ->setComment("Save process provenance information, e.g. for edmProvDump");
  desc.addUntracked<bool>("fakeNameForCrab", false)
      ->setComment(
          "Change the OutputModule name in the fwk job report to fake PoolOutputModule. This is needed to run on crab "
          "(and publish) till crab is fixed");
  desc.addUntracked<int>("autoFlush", -10000000)->setComment("Autoflush parameter for ROOT file");

  //Used by Workflow management for their own meta data
  edm::ParameterSetDescription dataSet;
  dataSet.setAllowAnything();
  desc.addUntracked<edm::ParameterSetDescription>("dataset", dataSet)
      ->setComment("PSet is only used by Data Operations and not by this module.");

  edm::ParameterSetDescription branchSet;
  branchSet.setAllowAnything();
  desc.add<edm::ParameterSetDescription>("branches", branchSet);
}
