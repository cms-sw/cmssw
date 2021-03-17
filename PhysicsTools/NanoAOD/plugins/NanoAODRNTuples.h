#ifndef PhysicsTools_NanoAOD_NanoAODRNTuples_h
#define PhysicsTools_NanoAOD_NanoAODRNTuples_h

#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"

#include "TFile.h"
#include <ROOT/RNTuple.hxx>
using ROOT::Experimental::RNTupleWriter;

#include "RNTupleFieldPtr.h"

class LumiNTuple {
public:
  LumiNTuple() = default;
  void createFields(TFile& file);
  void fill(const edm::LuminosityBlockID& id);
  void write();
private:
  std::unique_ptr<RNTupleWriter> m_ntuple;
  RNTupleFieldPtr<UInt_t> m_run;
  RNTupleFieldPtr<UInt_t> m_luminosityBlock;
};

#endif
