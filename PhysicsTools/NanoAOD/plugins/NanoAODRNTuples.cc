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

void LumiNTuple::createFields(TFile& file) {
  auto model = RNTupleModel::Create();
  m_run = RNTupleFieldPtr<UInt_t>("run", *model);
  m_luminosityBlock = RNTupleFieldPtr<UInt_t>("luminosityBlock", *model);
  // TODO use Append when we bump our RNTuple version:
  // m_ntuple = RNTupleWriter::Append(std::move(model), "LuminosityBlocks", file);
  m_ntuple = std::make_unique<RNTupleWriter>(std::move(model),
    std::make_unique<RPageSinkFile>("LuminosityBlocks", file, RNTupleWriteOptions())
  );
}

void LumiNTuple::fill(const edm::LuminosityBlockID& id) {
  m_run.fill(id.run());
  m_luminosityBlock.fill(id.value());
  m_ntuple->Fill();
}

void LumiNTuple::write() {
  m_ntuple.reset();
}
