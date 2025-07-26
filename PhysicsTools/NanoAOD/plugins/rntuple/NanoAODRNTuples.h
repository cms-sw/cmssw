#ifndef PhysicsTools_NanoAOD_NanoAODRNTuples_h
#define PhysicsTools_NanoAOD_NanoAODRNTuples_h

#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "TFile.h"
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleWriter.hxx>
using ROOT::RNTupleWriter;

#include "EventStringOutputFields.h"
#include "RNTupleFieldPtr.h"
#include "SummaryTableOutputFields.h"
#include "TableOutputFields.h"
#include "TriggerOutputFields.h"

class LumiNTuple {
public:
  LumiNTuple() = default;
  void fill(const edm::LuminosityBlockID& id, TFile& file);
  void finalizeWrite();

private:
  void createFields(const edm::LuminosityBlockID& id, TFile& file);
  std::unique_ptr<RNTupleWriter> m_ntuple;
  RNTupleFieldPtr<UInt_t> m_run;
  RNTupleFieldPtr<UInt_t> m_luminosityBlock;
};

class RunNTuple {
public:
  RunNTuple() = default;
  void registerToken(const edm::EDGetToken& token);
  void fill(const edm::RunForOutput& iRun, TFile& file);
  void finalizeWrite();

private:
  void createFields(const edm::RunForOutput& iRun, TFile& file);
  std::vector<edm::EDGetToken> m_tokens;
  std::unique_ptr<RNTupleWriter> m_ntuple;
  RNTupleFieldPtr<UInt_t> m_run;
  std::vector<SummaryTableOutputFields> m_tables;
};

class PSetNTuple {
public:
  PSetNTuple() = default;
  void fill(edm::pset::Registry* pset, TFile& file);
  void finalizeWrite();

private:
  using PSetType = std::pair<edm::ParameterSetID, edm::ParameterSetBlob>;
  RNTupleFieldPtr<PSetType> m_pset;
  void createFields(TFile& file);
  std::unique_ptr<RNTupleWriter> m_ntuple;
};

class MetadataNTuple {
public:
  MetadataNTuple() = default;
  void fill(const edm::ProcessHistoryRegistry& procHist, TFile& file);
  void finalizeWrite();

private:
  void createFields(TFile& file);
  RNTupleFieldPtr<edm::ProcessHistory> m_procHist;
  std::unique_ptr<RNTupleWriter> m_ntuple;
};

#endif
