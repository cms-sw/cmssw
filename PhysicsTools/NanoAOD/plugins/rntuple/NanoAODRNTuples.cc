#include "NanoAODRNTuples.h"

#include "DataFormats/NanoAOD/interface/MergeableCounterTable.h"
#include "FWCore/Framework/interface/RunForOutput.h"

#include <ROOT/RNTupleModel.hxx>
using ROOT::RNTupleModel;
#include <ROOT/RNTupleWriteOptions.hxx>
using ROOT::RNTupleWriteOptions;

#include "RNTupleFieldPtr.h"
#include "SummaryTableOutputFields.h"

void LumiNTuple::createFields(const edm::LuminosityBlockID& id, TFile& file) {
  auto model = RNTupleModel::Create();
  m_run = RNTupleFieldPtr<UInt_t>("run", "", *model);
  m_luminosityBlock = RNTupleFieldPtr<UInt_t>("luminosityBlock", "", *model);
  RNTupleWriteOptions options;
  options.SetCompression(file.GetCompressionSettings());
  m_ntuple = RNTupleWriter::Append(std::move(model), "LuminosityBlocks", file, options);
}

void LumiNTuple::fill(const edm::LuminosityBlockID& id, TFile& file) {
  if (!m_ntuple) {
    createFields(id, file);
  }
  m_run.fill(id.run());
  m_luminosityBlock.fill(id.value());
  m_ntuple->Fill();
}

void LumiNTuple::finalizeWrite() { m_ntuple.reset(); }

void RunNTuple::registerCounterTableToken(const edm::EDGetToken& token) { m_counterTableTokens.push_back(token); }

void RunNTuple::registerFlatTableToken(const edm::EDGetToken& token) { m_flatTableTokens.push_back(token); }

void RunNTuple::createFields(const edm::RunForOutput& iRun, TFile& file) {
  auto model = RNTupleModel::Create();
  m_run = RNTupleFieldPtr<UInt_t>("run", "", *model);

  edm::Handle<nanoaod::MergeableCounterTable> counterTableHandle;
  for (const auto& token : m_counterTableTokens) {
    iRun.getByToken(token, counterTableHandle);
    const nanoaod::MergeableCounterTable& tab = *counterTableHandle;
    m_counterTables.push_back(SummaryTableOutputFields(tab, *model));
  }

  edm::Handle<nanoaod::FlatTable> flatTableHandle;
  for (const auto& token : m_flatTableTokens) {
    iRun.getByToken(token, flatTableHandle);
    m_flatTables.add(token, *flatTableHandle);
  }
  m_flatTables.createFields(iRun, *model);

  RNTupleWriteOptions options;
  options.SetCompression(file.GetCompressionSettings());
  m_ntuple = RNTupleWriter::Append(std::move(model), "Runs", file, options);
}

void RunNTuple::fill(const edm::RunForOutput& iRun, TFile& file) {
  if (!m_ntuple) {
    createFields(iRun, file);
  }
  m_run.fill(iRun.id().run());

  edm::Handle<nanoaod::MergeableCounterTable> counterTableHandle;
  for (std::size_t i = 0; i < m_counterTableTokens.size(); i++) {
    iRun.getByToken(m_counterTableTokens.at(i), counterTableHandle);
    const nanoaod::MergeableCounterTable& tab = *counterTableHandle;
    m_counterTables.at(i).fill(tab);
  }

  m_flatTables.fill(iRun);

  m_ntuple->Fill();
}

void RunNTuple::finalizeWrite() { m_ntuple.reset(); }

void PSetNTuple::createFields(TFile& file) {
  auto model = RNTupleModel::Create();
  m_pset = RNTupleFieldPtr<PSetType>(edm::poolNames::idToParameterSetBlobsBranchName(), "", *model);

  RNTupleWriteOptions options;
  options.SetCompression(file.GetCompressionSettings());
  m_ntuple = RNTupleWriter::Append(std::move(model), edm::poolNames::parameterSetsTreeName(), file, options);
}

void PSetNTuple::fill(edm::pset::Registry* pset, TFile& file) {
  if (!pset) {
    throw cms::Exception("LogicError", "null edm::pset::Registry::Instance pointer");
  }
  if (!m_ntuple) {
    createFields(file);
  }
  for (const auto& ps : *pset) {
    std::string psString;
    ps.second.toString(psString);
    edm::ParameterSetBlob psBlob(psString);
    m_pset.fill(std::make_pair(ps.first, psBlob));
    m_ntuple->Fill();
  }
}

void PSetNTuple::finalizeWrite() { m_ntuple.reset(); }

void MetadataNTuple::createFields(TFile& file) {
  auto model = RNTupleModel::Create();
  m_procHist = RNTupleFieldPtr<edm::ProcessHistory>(edm::poolNames::processHistoryBranchName(), "", *model);
  RNTupleWriteOptions options;
  options.SetCompression(file.GetCompressionSettings());
  m_ntuple = RNTupleWriter::Append(std::move(model), edm::poolNames::metaDataTreeName(), file, options);
}

void MetadataNTuple::fill(const edm::ProcessHistoryRegistry& procHist, TFile& file) {
  if (!m_ntuple) {
    createFields(file);
  }
  for (const auto& ph : procHist) {
    m_procHist.fill(ph.second);
    m_ntuple->Fill();
  }
}

void MetadataNTuple::finalizeWrite() { m_ntuple.reset(); }
