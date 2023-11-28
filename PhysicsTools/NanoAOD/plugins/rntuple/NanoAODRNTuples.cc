#include "NanoAODRNTuples.h"

#include "DataFormats/NanoAOD/interface/MergeableCounterTable.h"
#include "FWCore/Framework/interface/RunForOutput.h"

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RPageStorageFile.hxx>
using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::RNTupleWriteOptions;
using ROOT::Experimental::RNTupleWriter;
using ROOT::Experimental::Detail::RPageSinkFile;

#include "RNTupleFieldPtr.h"
#include "SummaryTableOutputFields.h"

void LumiNTuple::createFields(const edm::LuminosityBlockID& id, TFile& file) {
  auto model = RNTupleModel::Create();
  m_run = RNTupleFieldPtr<UInt_t>("run", "", *model);
  m_luminosityBlock = RNTupleFieldPtr<UInt_t>("luminosityBlock", "", *model);
  // TODO use Append when we bump our RNTuple version:
  // m_ntuple = RNTupleWriter::Append(std::move(model), "LuminosityBlocks", file);
  RNTupleWriteOptions options;
  options.SetCompression(file.GetCompressionSettings());
  m_ntuple = std::make_unique<RNTupleWriter>(std::move(model),
                                             std::make_unique<RPageSinkFile>("LuminosityBlocks", file, options));
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

void RunNTuple::registerToken(const edm::EDGetToken& token) { m_tokens.push_back(token); }

void RunNTuple::createFields(const edm::RunForOutput& iRun, TFile& file) {
  auto model = RNTupleModel::Create();
  m_run = RNTupleFieldPtr<UInt_t>("run", "", *model);

  edm::Handle<nanoaod::MergeableCounterTable> handle;
  for (const auto& token : m_tokens) {
    iRun.getByToken(token, handle);
    const nanoaod::MergeableCounterTable& tab = *handle;
    m_tables.push_back(SummaryTableOutputFields(tab, *model));
  }

  // TODO use Append when we bump our RNTuple version
  RNTupleWriteOptions options;
  options.SetCompression(file.GetCompressionSettings());
  m_ntuple = std::make_unique<RNTupleWriter>(std::move(model), std::make_unique<RPageSinkFile>("Runs", file, options));
}

void RunNTuple::fill(const edm::RunForOutput& iRun, TFile& file) {
  if (!m_ntuple) {
    createFields(iRun, file);
  }
  m_run.fill(iRun.id().run());
  edm::Handle<nanoaod::MergeableCounterTable> handle;
  for (std::size_t i = 0; i < m_tokens.size(); i++) {
    iRun.getByToken(m_tokens.at(i), handle);
    const nanoaod::MergeableCounterTable& tab = *handle;
    m_tables.at(i).fill(tab);
  }
  m_ntuple->Fill();
}

void RunNTuple::finalizeWrite() { m_ntuple.reset(); }

void PSetNTuple::createFields(TFile& file) {
  // use a collection to emulate std::pair
  auto pairModel = RNTupleModel::Create();
  m_psetId = RNTupleFieldPtr<std::string>("first", "", *pairModel);
  m_psetBlob = RNTupleFieldPtr<std::string>("second", "", *pairModel);
  auto model = RNTupleModel::Create();
  m_collection = model->MakeCollection(edm::poolNames::idToParameterSetBlobsBranchName(), std::move(pairModel));
  // TODO use Append when we bump our RNTuple version
  RNTupleWriteOptions options;
  options.SetCompression(file.GetCompressionSettings());
  m_ntuple = std::make_unique<RNTupleWriter>(
      std::move(model), std::make_unique<RPageSinkFile>(edm::poolNames::parameterSetsTreeName(), file, options));
}

void PSetNTuple::fill(edm::pset::Registry* pset, TFile& file) {
  if (!pset) {
    throw cms::Exception("LogicError", "null edm::pset::Registry::Instance pointer");
  }
  if (!m_ntuple) {
    createFields(file);
  }
  for (const auto& ps : *pset) {
    std::ostringstream oss;
    oss << ps.first;
    m_psetId.fill(oss.str());
    m_psetBlob.fill(ps.second.toString());
    m_collection->Fill();
    m_ntuple->Fill();
  }
}

void PSetNTuple::finalizeWrite() { m_ntuple.reset(); }

// TODO blocked on RNTuple typedef member field support
void MetadataNTuple::createFields(TFile& file) {
  auto procHistModel = RNTupleModel::Create();
  // ProcessHistory.transients_.phid_ replacement
  m_phId = RNTupleFieldPtr<std::string>("transients_phid_", "", *procHistModel);
  auto model = RNTupleModel::Create();
  m_procHist = model->MakeCollection(edm::poolNames::processHistoryBranchName(), std::move(procHistModel));
  RNTupleWriteOptions options;
  options.SetCompression(file.GetCompressionSettings());
  m_ntuple = std::make_unique<RNTupleWriter>(
      std::move(model), std::make_unique<RPageSinkFile>(edm::poolNames::metaDataTreeName(), file, options));
}

void MetadataNTuple::fill(const edm::ProcessHistoryRegistry& procHist, TFile& file) {
  if (!m_ntuple) {
    createFields(file);
  }
  for (const auto& ph : procHist) {
    std::string phid;
    ph.second.id().toString(phid);
    m_phId.fill(phid);
    m_procHist->Fill();
  }
  m_ntuple->Fill();
}

void MetadataNTuple::finalizeWrite() { m_ntuple.reset(); }
