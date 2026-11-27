#include "NanoAODRNTuples.h"

#include "DataFormats/NanoAOD/interface/MergeableCounterTable.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>

#include "RNTupleFieldPtr.h"
#include "SummaryTableOutputFields.h"

using ROOT::RNTupleModel;
using ROOT::RNTupleWriteOptions;
using ROOT::RNTupleWriter;

namespace {
  // Every ntuple in the file inherits the file's compression and takes the ROOT defaults otherwise.
  RNTupleWriteOptions writeOptions(const TFile& file) {
    RNTupleWriteOptions options;
    options.SetCompression(file.GetCompressionSettings());
    return options;
  }
}  // anonymous namespace

void LumiNTuple::registerCounterTableToken(const edm::EDGetToken& token) { m_counterTableTokens.push_back(token); }

void LumiNTuple::registerFlatTableToken(const edm::EDGetToken& token) { m_flatTableTokens.push_back(token); }

void LumiNTuple::createFields(const edm::LuminosityBlockForOutput& iLumi, TFile& file) {
  auto model = RNTupleModel::Create();
  m_run = RNTupleFieldPtr<std::uint32_t>("run", "", *model);
  m_luminosityBlock = RNTupleFieldPtr<std::uint32_t>("luminosityBlock", "", *model);

  edm::Handle<nanoaod::MergeableCounterTable> counterTableHandle;
  for (const auto& token : m_counterTableTokens) {
    iLumi.getByToken(token, counterTableHandle);
    m_counterTables.emplace_back(*counterTableHandle, *model);
  }

  edm::Handle<nanoaod::FlatTable> flatTableHandle;
  for (const auto& token : m_flatTableTokens) {
    iLumi.getByToken(token, flatTableHandle);
    m_flatTables.add(token, *flatTableHandle);
  }
  m_flatTables.createFields(iLumi, *model);
  if (m_flatProjections) {
    m_flatTables.addProjections(*model);
  }

  // Collection buffers can only be bound once the schema is frozen, and the writer takes ownership
  // of the model, so this has to happen here rather than after Append().
  model->Freeze();
  m_flatTables.bind(model->GetDefaultEntry());
  m_ntuple = RNTupleWriter::Append(std::move(model), "LuminosityBlocks", file, writeOptions(file));
}

void LumiNTuple::fill(const edm::LuminosityBlockForOutput& iLumi, TFile& file) {
  if (!m_ntuple) {
    createFields(iLumi, file);
  }
  m_run.fill(iLumi.id().run());
  m_luminosityBlock.fill(iLumi.id().value());

  edm::Handle<nanoaod::MergeableCounterTable> counterTableHandle;
  for (std::size_t i = 0; i < m_counterTableTokens.size(); i++) {
    iLumi.getByToken(m_counterTableTokens[i], counterTableHandle);
    m_counterTables[i].fill(*counterTableHandle);
  }

  m_flatTables.fill(iLumi);

  m_ntuple->Fill();
}

void LumiNTuple::finalizeWrite() { m_ntuple.reset(); }

void RunNTuple::registerCounterTableToken(const edm::EDGetToken& token) { m_counterTableTokens.push_back(token); }

void RunNTuple::registerFlatTableToken(const edm::EDGetToken& token) { m_flatTableTokens.push_back(token); }

void RunNTuple::createFields(const edm::RunForOutput& iRun, TFile& file) {
  auto model = RNTupleModel::Create();
  m_run = RNTupleFieldPtr<std::uint32_t>("run", "", *model);

  edm::Handle<nanoaod::MergeableCounterTable> counterTableHandle;
  for (const auto& token : m_counterTableTokens) {
    iRun.getByToken(token, counterTableHandle);
    m_counterTables.emplace_back(*counterTableHandle, *model);
  }

  edm::Handle<nanoaod::FlatTable> flatTableHandle;
  for (const auto& token : m_flatTableTokens) {
    iRun.getByToken(token, flatTableHandle);
    m_flatTables.add(token, *flatTableHandle);
  }
  m_flatTables.createFields(iRun, *model);
  if (m_flatProjections) {
    m_flatTables.addProjections(*model);
  }

  // See LumiNTuple::createFields for why the bind sits between the freeze and the Append().
  model->Freeze();
  m_flatTables.bind(model->GetDefaultEntry());
  m_ntuple = RNTupleWriter::Append(std::move(model), "Runs", file, writeOptions(file));
}

void RunNTuple::fill(const edm::RunForOutput& iRun, TFile& file) {
  if (!m_ntuple) {
    createFields(iRun, file);
  }
  m_run.fill(iRun.id().run());

  edm::Handle<nanoaod::MergeableCounterTable> counterTableHandle;
  for (std::size_t i = 0; i < m_counterTableTokens.size(); i++) {
    iRun.getByToken(m_counterTableTokens[i], counterTableHandle);
    m_counterTables[i].fill(*counterTableHandle);
  }

  m_flatTables.fill(iRun);

  m_ntuple->Fill();
}

void RunNTuple::finalizeWrite() { m_ntuple.reset(); }

namespace rntupleprovenance {

  void writeParameterSets(TFile& file) {
    using PSetType = std::pair<edm::ParameterSetID, edm::ParameterSetBlob>;

    auto model = RNTupleModel::Create();
    auto psets = RNTupleFieldPtr<PSetType>(edm::poolNames::idToParameterSetBlobsBranchName(), "", *model);
    // The writer commits when it goes out of scope at the end of this function.
    auto ntuple =
        RNTupleWriter::Append(std::move(model), edm::poolNames::parameterSetsTreeName(), file, writeOptions(file));

    for (const auto& ps : *edm::pset::Registry::instance()) {
      std::string psString;
      ps.second.toString(psString);
      psets.fill(std::make_pair(ps.first, edm::ParameterSetBlob(psString)));
      ntuple->Fill();
    }
  }

  void writeProcessHistory(const edm::ProcessHistoryRegistry& procHist, TFile& file) {
    auto model = RNTupleModel::Create();
    auto history = RNTupleFieldPtr<edm::ProcessHistory>(edm::poolNames::processHistoryBranchName(), "", *model);
    auto ntuple = RNTupleWriter::Append(std::move(model), edm::poolNames::metaDataTreeName(), file, writeOptions(file));

    for (const auto& ph : procHist) {
      history.fill(ph.second);
      ntuple->Fill();
    }
  }

}  // namespace rntupleprovenance
