// -*- C++ -*-
//
// Package:     PhysicsTools/NanoAODOutput
// Class  :     NanoAODRNTupleOutputModule
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Max Orok
//         Created:  Wed, 13 Jan 2021 14:21:41 GMT
//

#include <cstdint>
#include <string>

#include <ROOT/REntry.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include "Compression.h"
#include "TObjString.h"

#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "DataFormats/NanoAOD/interface/UniqueString.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"

#include "EventStringOutputFields.h"
#include "NanoAODRNTuples.h"
#include "RNTupleFieldPtr.h"
#include "TriggerOutputFields.h"

using ROOT::RNTupleModel;
using ROOT::RNTupleWriteOptions;
using ROOT::RNTupleWriter;

class NanoAODRNTupleOutputModule : public edm::one::OutputModule<> {
public:
  NanoAODRNTupleOutputModule(edm::ParameterSet const& pset);
  ~NanoAODRNTupleOutputModule() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void openFile(edm::FileBlock const&) override;
  bool isFileOpen() const override;
  void write(edm::EventForOutput const& e) override;
  void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override;
  void writeRun(edm::RunForOutput const&) override;
  void reallyCloseFile() override;
  void writeProvenance();

  void initializeNTuple(edm::EventForOutput const& e);
  // Create the entry the Events ntuple is filled through and point every field at its value. The
  // Events model is bare, so there is no default entry to fall back on, and a schema update
  // invalidates the entry, so this runs again after each one.
  void bindEntry();

  std::string m_fileName;
  std::string m_logicalFileName;
  ROOT::RCompressionSetting::EAlgorithm::EValues m_compressionAlgorithm;
  int m_compressionLevel;
  bool m_writeProvenance;
  edm::ProcessHistoryRegistry m_processHistoryRegistry;
  edm::JobReport::Token m_jrToken;

  std::unique_ptr<TFile> m_file;
  std::unique_ptr<RNTupleWriter> m_ntuple;
  std::unique_ptr<ROOT::REntry> m_entry;
  std::vector<edm::EDGetToken> m_eventTableTokens;
  TableCollectionSet m_tables;
  std::vector<TriggerOutputFields> m_triggers;
  EventStringOutputFields m_evstrings;

  class CommonEventFields {
  public:
    void createFields(RNTupleModel& model) {
      m_run = RNTupleFieldPtr<std::uint32_t>("run", "", model);
      m_luminosityBlock = RNTupleFieldPtr<std::uint32_t>("luminosityBlock", "", model);
      m_event = RNTupleFieldPtr<std::uint64_t>("event", "", model);
      m_bunchCrossing = RNTupleFieldPtr<std::uint32_t>("bunchCrossing", "", model);
      m_orbitNumber = RNTupleFieldPtr<std::uint32_t>("orbitNumber", "", model);
    }
    void bind(ROOT::REntry& entry) const {
      m_run.bind(entry);
      m_luminosityBlock.bind(entry);
      m_event.bind(entry);
      m_bunchCrossing.bind(entry);
      m_orbitNumber.bind(entry);
    }
    void fill(const edm::EventAuxiliary& aux) {
      m_run.fill(aux.id().run());
      m_luminosityBlock.fill(aux.id().luminosityBlock());
      m_event.fill(aux.id().event());
      // EventAuxiliary reports both of these as a signed int, using -1 for "not available" (as in
      // simulation). NanoAOD has always published them unsigned, so an unavailable value shows up
      // as 0xffffffff; keep that rather than inventing a representation the TTree output lacks.
      m_bunchCrossing.fill(static_cast<std::uint32_t>(aux.bunchCrossing()));
      m_orbitNumber.fill(static_cast<std::uint32_t>(aux.orbitNumber()));
    }

  private:
    RNTupleFieldPtr<std::uint32_t> m_run;
    RNTupleFieldPtr<std::uint32_t> m_luminosityBlock;
    RNTupleFieldPtr<std::uint64_t> m_event;
    RNTupleFieldPtr<std::uint32_t> m_bunchCrossing;
    RNTupleFieldPtr<std::uint32_t> m_orbitNumber;
  } m_commonFields;

  LumiNTuple m_lumi;
  RunNTuple m_run;

  std::vector<std::pair<std::string, edm::EDGetToken>> m_nanoMetadata;

  std::vector<std::string> m_noSplitFields;
  // Whether the members of the grouped fields also get the flat TTree names; see rntupleprojection.
  bool m_flatProjections;
  ROOT::RNTupleWriteOptions m_writeOptions;
};

namespace {
  // The four algorithms the TTree NanoAODOutputModule accepts. RNTuple compresses its pages through
  // the same ROOT machinery, so it supports all of them; parse here so a bad name fails at
  // construction rather than when the first output file is opened.
  ROOT::RCompressionSetting::EAlgorithm::EValues compressionAlgorithm(const std::string& name) {
    using EAlgorithm = ROOT::RCompressionSetting::EAlgorithm;
    if (name == "ZLIB") {
      return EAlgorithm::kZLIB;
    } else if (name == "LZMA") {
      return EAlgorithm::kLZMA;
    } else if (name == "ZSTD") {
      return EAlgorithm::kZSTD;
    } else if (name == "LZ4") {
      return EAlgorithm::kLZ4;
    }
    throw cms::Exception("Configuration")
        << "NanoAODRNTupleOutputModule configured with unknown compression algorithm '" << name << "'\n"
        << "Allowed compression algorithms are ZLIB, LZMA, ZSTD, and LZ4\n";
  }

  ROOT::RNTupleWriteOptions writeOptions(edm::ParameterSet const& iConfig) {
    ROOT::RNTupleWriteOptions options;

    options.SetApproxZippedClusterSize(iConfig.getUntrackedParameter<unsigned long long>("approxZippedClusterSize"));

    options.SetMaxUnzippedClusterSize(iConfig.getUntrackedParameter<unsigned long long>("maxUnzippedClusterSize"));

    options.SetInitialUnzippedPageSize(iConfig.getUntrackedParameter<unsigned long long>("initialUnzippedPageSize"));
    options.SetMaxUnzippedPageSize(iConfig.getUntrackedParameter<unsigned long long>("maxUnzippedPageSize"));
    options.SetPageBufferBudget(iConfig.getUntrackedParameter<unsigned long long>("pageBufferBudget"));
    options.SetUseBufferedWrite(iConfig.getUntrackedParameter<bool>("useBufferedWrite"));
    options.SetUseDirectIO(iConfig.getUntrackedParameter<bool>("useDirectIO"));
    return options;
  }
}  // namespace

NanoAODRNTupleOutputModule::NanoAODRNTupleOutputModule(edm::ParameterSet const& pset)
    : edm::one::OutputModuleBase::OutputModuleBase(pset),
      edm::one::OutputModule<>(pset),
      m_fileName(pset.getUntrackedParameter<std::string>("fileName")),
      m_logicalFileName(pset.getUntrackedParameter<std::string>("logicalFileName")),
      m_compressionAlgorithm(compressionAlgorithm(pset.getUntrackedParameter<std::string>("compressionAlgorithm"))),
      m_compressionLevel(pset.getUntrackedParameter<int>("compressionLevel")),
      m_writeProvenance(pset.getUntrackedParameter<bool>("saveProvenance", true)),
      m_noSplitFields{pset.getUntrackedParameter<std::vector<std::string>>("noSplitFields")},
      m_flatProjections(pset.getUntrackedParameter<bool>("flatProjections")),
      m_writeOptions(writeOptions(pset.getUntrackedParameterSet("rntupleWriteOptions"))) {
  m_lumi.setFlatProjections(m_flatProjections);
  m_run.setFlatProjections(m_flatProjections);
}

void NanoAODRNTupleOutputModule::writeLuminosityBlock(edm::LuminosityBlockForOutput const& iLumi) {
  edm::Service<edm::JobReport> jr;
  jr->reportLumiSection(m_jrToken, iLumi.id().run(), iLumi.id().value());
  m_lumi.fill(iLumi, *m_file);
  m_processHistoryRegistry.registerProcessHistory(iLumi.processHistory());
}

void NanoAODRNTupleOutputModule::writeRun(edm::RunForOutput const& iRun) {
  edm::Service<edm::JobReport> jr;
  jr->reportRunNumber(m_jrToken, iRun.id().run());

  m_run.fill(iRun, *m_file);

  edm::Handle<nanoaod::UniqueString> hstring;
  for (const auto& p : m_nanoMetadata) {
    iRun.getByToken(p.second, hstring);
    TObjString* tos = dynamic_cast<TObjString*>(m_file->Get(p.first.c_str()));
    if (tos) {
      if (hstring->str() != tos->GetString()) {
        throw cms::Exception("LogicError", "Inconsistent nanoMetadata " + p.first + " (" + hstring->str() + ")");
      }
    } else {
      auto ostr = std::make_unique<TObjString>(hstring->str().c_str());
      m_file->WriteTObject(ostr.release(), p.first.c_str());
    }
  }
  m_processHistoryRegistry.registerProcessHistory(iRun.processHistory());
}

// The Events RNTuple is only created on the first event, so the file, not the writer, tracks
// whether output is open: otherwise a job writing no events would never be closed.
bool NanoAODRNTupleOutputModule::isFileOpen() const { return nullptr != m_file.get(); }

void NanoAODRNTupleOutputModule::openFile(edm::FileBlock const&) {
  m_file = std::make_unique<TFile>(m_fileName.c_str(), "RECREATE", "", m_compressionLevel);
  edm::Service<edm::JobReport> jr;
  cms::Digest branchHash;
  m_jrToken = jr->outputFileOpened(m_fileName,
                                   m_logicalFileName,
                                   std::string(),
                                   "NanoAODRNTupleOutputModule",
                                   description().moduleLabel(),
                                   edm::createGlobalIdentifier(),
                                   std::string(),
                                   branchHash.digest().toString(),
                                   std::vector<std::string>());

  m_file->SetCompressionAlgorithm(m_compressionAlgorithm);
  // Every RNTuple in the file takes its compression from the file, so this must follow the setter.
  m_writeOptions.SetCompression(m_file->GetCompressionSettings());

  // Sorting the kept products by class only needs their descriptions, so it happens here; what the
  // Events model looks like does depend on the payloads and is left to initializeNTuple().
  const auto& keeps = keptProducts();
  for (const auto& keep : keeps[edm::InRun]) {
    if (keep.first->className() == "nanoaod::MergeableCounterTable") {
      m_run.registerCounterTableToken(keep.second);
    } else if (keep.first->className() == "nanoaod::UniqueString" && keep.first->moduleLabel() == "nanoMetadata") {
      m_nanoMetadata.emplace_back(keep.first->productInstanceName(), keep.second);
    } else if (keep.first->className() == "nanoaod::FlatTable") {
      m_run.registerFlatTableToken(keep.second);
    } else {
      throw cms::Exception(
          "Configuration",
          "NanoAODRNTupleOutputModule cannot handle class " + keep.first->className() + " in Run RNTuple");
    }
  }

  // nanoaod::UniqueString is deliberately absent here: UniqueStringProducer only produces InRun, so
  // a lumi-level one would need handling that cannot be exercised, and the throw below says so.
  for (const auto& keep : keeps[edm::InLumi]) {
    if (keep.first->className() == "nanoaod::MergeableCounterTable") {
      m_lumi.registerCounterTableToken(keep.second);
    } else if (keep.first->className() == "nanoaod::FlatTable") {
      m_lumi.registerFlatTableToken(keep.second);
    } else {
      throw cms::Exception(
          "Configuration",
          "NanoAODRNTupleOutputModule cannot handle class " + keep.first->className() + " in LuminosityBlock RNTuple");
    }
  }

  for (const auto& keep : keeps[edm::InEvent]) {
    if (keep.first->className() == "nanoaod::FlatTable") {
      m_eventTableTokens.push_back(keep.second);
    } else if (keep.first->className() == "edm::TriggerResults") {
      m_triggers.emplace_back(keep.first->processName(), keep.second);
    } else if (keep.first->className() == "std::basic_string<char,std::char_traits<char> >" &&
               keep.first->productInstanceName() == "genModel") {
      m_evstrings.registerToken(keep.second);
    } else {
      throw cms::Exception("Configuration",
                           "NanoAODRNTupleOutputModule cannot handle class " + keep.first->className());
    }
  }
}

namespace {
  void noSplitField(ROOT::RFieldBase& iField) {
    auto const& typeName = iField.GetTypeName();
    if (typeName == "std::uint16_t") {
      iField.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kUInt16}});
    } else if (typeName == "std::uint32_t") {
      iField.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kUInt32}});
    } else if (typeName == "std::uint64_t") {
      iField.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kUInt64}});
    } else if (typeName == "std::int16_t") {
      iField.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kInt16}});
    } else if (typeName == "std::int32_t") {
      iField.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kInt32}});
    } else if (typeName == "std::int64_t") {
      iField.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kInt64}});
    } else if (typeName == "float") {
      iField.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kReal32}});
    } else if (typeName == "double") {
      iField.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kReal64}});
    }
  }
  void applyNoSplitToSubFields(ROOT::RFieldBase& iField) {
    for (auto& subfield : iField) {
      noSplitField(subfield);
      applyNoSplitToSubFields(subfield);
    }
  }
}  // namespace

void NanoAODRNTupleOutputModule::initializeNTuple(edm::EventForOutput const& iEvent) {
  // RNTuple needs the whole schema before the first Fill, and which fields a FlatTable needs is
  // only visible from its contents, so the Events model is built on the first event.
  //
  // Bare, i.e. with no default entry of its own: ROOT refuses to add a subfield to an untyped
  // record of a non-bare model, and that is how a trigger path first seen in a later run gets a
  // field (TriggerRecordFields::update). Every field is therefore bound to m_entry by hand.
  auto model = RNTupleModel::CreateBare();
  m_commonFields.createFields(*model);

  edm::Handle<nanoaod::FlatTable> handle;
  for (const auto& token : m_eventTableTokens) {
    iEvent.getByToken(token, handle);
    m_tables.add(token, *handle);
  }
  m_tables.createFields(iEvent, *model);
  for (auto& trigger : m_triggers) {
    trigger.createFields(iEvent, *model);
  }
  m_evstrings.createFields(*model);

  if (m_noSplitFields.size() == 1 and m_noSplitFields[0] == "all") {
    for (auto const& topName : model->GetFieldNames()) {
      auto& field = model->GetMutableField(topName);
      noSplitField(field);
      applyNoSplitToSubFields(field);
    }
  } else {
    for (auto const& name : m_noSplitFields) {
      auto& field = model->GetMutableField(name);
      noSplitField(field);
    }
  }

  // After the noSplit pass, so that a projection never lands on a source field whose column
  // representation is about to be pinned, and so that the pass itself only ever sees real fields.
  if (m_flatProjections) {
    m_tables.addProjections(*model);
    for (auto& trigger : m_triggers) {
      trigger.addProjections(*model);
    }
  }

  // Model needs to be frozen before we bind buffers
  model->Freeze();

  m_ntuple = RNTupleWriter::Append(std::move(model), "Events", *m_file, m_writeOptions);
  bindEntry();
}

void NanoAODRNTupleOutputModule::bindEntry() {
  const auto& model = m_ntuple->GetModel();
  m_entry = model.CreateBareEntry();
  m_commonFields.bind(*m_entry);
  m_tables.bind(*m_entry);
  for (auto& trigger : m_triggers) {
    trigger.bind(*m_entry, model);
  }
  m_evstrings.bind(*m_entry);
}

void NanoAODRNTupleOutputModule::write(edm::EventForOutput const& iEvent) {
  if (!m_ntuple) {
    initializeNTuple(iEvent);
  }

  edm::Service<edm::JobReport> jr;
  jr->eventWrittenToFile(m_jrToken, iEvent.id().run(), iEvent.id().event());

  // A run boundary can bring a menu that the schema was not built from, and a trigger record that
  // grows to fit it leaves every entry of the old schema invalid.
  bool schemaChanged = false;
  for (auto& trigger : m_triggers) {
    schemaChanged |= trigger.updateForRun(iEvent, *m_ntuple, m_flatProjections);
  }
  if (schemaChanged) {
    bindEntry();
  }

  m_commonFields.fill(iEvent.eventAuxiliary());
  m_tables.fill(iEvent);
  for (auto& trigger : m_triggers) {
    trigger.fill(iEvent);
  }
  m_evstrings.fill(iEvent);
  m_ntuple->Fill(*m_entry);
  m_processHistoryRegistry.registerProcessHistory(iEvent.processHistory());
}

void NanoAODRNTupleOutputModule::reallyCloseFile() {
  if (m_writeProvenance) {
    writeProvenance();
  }
  // write ntuple to disk by calling the RNTupleWriter destructor, after letting go of the entry
  // that borrows the model the writer owns
  m_entry.reset();
  m_ntuple.reset();
  m_lumi.finalizeWrite();
  m_run.finalizeWrite();
  m_file->Write();
  m_file->Close();
  m_file.reset();

  edm::Service<edm::JobReport> jr;
  jr->outputFileClosed(m_jrToken);
}

void NanoAODRNTupleOutputModule::writeProvenance() {
  rntupleprovenance::writeParameterSets(*m_file);
  rntupleprovenance::writeProcessHistory(m_processHistoryRegistry, *m_file);
}

void NanoAODRNTupleOutputModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.addUntracked<std::string>("fileName");
  desc.addUntracked<std::string>("logicalFileName", "");
  desc.addUntracked<int>("compressionLevel", 9)->setComment("ROOT compression level of output file.");
  desc.addUntracked<std::string>("compressionAlgorithm", "ZLIB")
      ->setComment(
          "Algorithm used to "
          "compress data in the ROOT output file, allowed values are ZLIB, LZMA, ZSTD, and LZ4");
  desc.addUntracked<std::vector<std::string>>("noSplitFields", {})
      ->setComment("Name of fields to avoid the standard ROOT split optimization.");
  desc.addUntracked<bool>("flatProjections", true)
      ->setComment(
          "Also give every member of a grouped field the flat name the TTree output uses for it: "
          "GenJet_pt beside GenJet.pt, HLT_IsoMu24 beside HLT.IsoMu24. These are projected fields, "
          "i.e. second names for the same columns, so they add nothing to the file but their schema "
          "entry. Set false to write the grouped names only.");
  {
    edm::ParameterSetDescription optimizations;

    ROOT::RNTupleWriteOptions ops;
    optimizations.addUntracked<unsigned long long>("approxZippedClusterSize", ops.GetApproxZippedClusterSize())
        ->setComment("Approximation of the target compressed cluster size");
    optimizations.addUntracked<unsigned long long>("maxUnzippedClusterSize", ops.GetMaxUnzippedClusterSize())
        ->setComment("Memory limit for committing a cluster. High compression leads to high IO buffer size.");

    optimizations.addUntracked<unsigned long long>("initialUnzippedPageSize", ops.GetInitialUnzippedPageSize())
        ->setComment("Initially, columns start with a page of this size (bytes).");
    optimizations.addUntracked<unsigned long long>("maxUnzippedPageSize", ops.GetMaxUnzippedPageSize())
        ->setComment("Pages can grow only to the given limit (bytes).");
    optimizations.addUntracked<unsigned long long>("pageBufferBudget", 0)
        ->setComment(
            "The maximum size that the sum of all page buffers used for writing into a persistent sink are allowed "
            "to use."
            " If set to zero, RNTuple will auto-adjust the budget based on the value of 'approxZippedClusterSize'."
            " If set manually, the size needs to be large enough to hold all initial page buffers.");

    optimizations.addUntracked<bool>("useBufferedWrite", ops.GetUseBufferedWrite())
        ->setComment(
            "Turn on use of buffered writing. This buffers compressed pages in memory, reorders them to keep pages "
            "of the same column adjacent, and coalesces the writes when committing a cluster.");
    optimizations.addUntracked<bool>("useDirectIO", ops.GetUseDirectIO())
        ->setComment(
            "Set use of direct IO. this introduces alignment requirements that may vary between filesystems and "
            "platforms");
    desc.addUntracked("rntupleWriteOptions", optimizations)
        ->setComment("Options to control RNTuple specific output features.");
  }
  desc.addUntracked<bool>("saveProvenance", true)
      ->setComment("Save process provenance information, e.g. for edmProvDump");
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

DEFINE_FWK_MODULE(NanoAODRNTupleOutputModule);
