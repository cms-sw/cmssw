#include "PhysicsTools/NanoAOD/plugins/NanoAODRNTuples.h"

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RPageStorageFile.hxx>
using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::RNTupleWriter;
using ROOT::Experimental::Detail::RPageSinkFile;
using ROOT::Experimental::RNTupleWriteOptions;

#include "RNTupleFieldPtr.h"

void LumiNTuple::createFields(const edm::LuminosityBlockID& id, TFile& file) {
  auto model = RNTupleModel::Create();
  m_run = RNTupleFieldPtr<UInt_t>("run", *model);
  m_luminosityBlock = RNTupleFieldPtr<UInt_t>("luminosityBlock", *model);
  // TODO use Append when we bump our RNTuple version:
  // m_ntuple = RNTupleWriter::Append(std::move(model), "LuminosityBlocks", file);
  RNTupleWriteOptions options;
  options.SetCompression(file.GetCompressionSettings());
  m_ntuple = std::make_unique<RNTupleWriter>(std::move(model),
    std::make_unique<RPageSinkFile>("LuminosityBlocks", file, options)
  );
}

void LumiNTuple::fill(const edm::LuminosityBlockID& id, TFile& file) {
  if (!m_ntuple) {
    createFields(id, file);
  }
  m_run.fill(id.run());
  m_luminosityBlock.fill(id.value());
  m_ntuple->Fill();
}

void LumiNTuple::finalizeWrite() {
  m_ntuple.reset();
}

// TODO SummaryTableOutput fields
void RunNTuple::createFields(const edm::RunForOutput& iRun, TFile& file) {
  auto model = RNTupleModel::Create();
  m_run = RNTupleFieldPtr<UInt_t>("run", *model);
  // TODO use Append when we bump our RNTuple version
  RNTupleWriteOptions options;
  options.SetCompression(file.GetCompressionSettings());
  m_ntuple = std::make_unique<RNTupleWriter>(std::move(model),
    std::make_unique<RPageSinkFile>("Runs", file, options)
  );
}

void RunNTuple::fill(const edm::RunForOutput& iRun, TFile& file) {
  if (!m_ntuple) {
    createFields(iRun, file);
  }
  m_run.fill(iRun.id().run());
  // todo fill SummaryTableOutputs
  m_ntuple->Fill();
}

void RunNTuple::finalizeWrite() {
  m_ntuple.reset();
}

void PSetNTuple::createFields(TFile& file) {
  auto model = RNTupleModel::Create();
  m_pset = RNTupleFieldPtr<PSetType>(edm::poolNames::idToParameterSetBlobsBranchName(), *model);
  // TODO use Append when we bump our RNTuple version
  RNTupleWriteOptions options;
  options.SetCompression(file.GetCompressionSettings());
  m_ntuple = std::make_unique<RNTupleWriter>(std::move(model),
    std::make_unique<RPageSinkFile>(edm::poolNames::parameterSetsTreeName(), file, options)
  );
}

void PSetNTuple::fill(edm::pset::Registry* pset, TFile& file) {
  if (!pset) {
    throw cms::Exception("LogicError", "null edm::pset::Registry::Instance pointer");
  }
  if (!m_ntuple) {
    createFields(file);
  }
  PSetType entry;
  for (const auto& ps: *pset) {
    // TODO fix string pair hack
    //entry.first = ps.first;
    //entry.second.pset() = ps.second.toString();
    std::ostringstream oss;
    oss << ps.first;
    entry.first = oss.str();
    entry.second = ps.second.toString();
    m_pset.fill(entry);
    m_ntuple->Fill();
  }
}

void PSetNTuple::finalizeWrite() {
  m_ntuple.reset();
}
