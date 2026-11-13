#ifndef PhysicsTools_NanoAOD_NanoAODRNTuples_h
#define PhysicsTools_NanoAOD_NanoAODRNTuples_h

#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "TFile.h"
#include <ROOT/RNTupleWriter.hxx>

#include "RNTupleFieldPtr.h"
#include "SummaryTableOutputFields.h"
#include "TableOutputFields.h"

// The LuminosityBlocks and Runs ntuples are created on their first fill, because their schema is
// only known once the first payload is in hand, and committed by finalizeWrite().

class LumiNTuple {
public:
  LumiNTuple() = default;
  void registerCounterTableToken(const edm::EDGetToken& token);
  void registerFlatTableToken(const edm::EDGetToken& token);
  // Whether the collections also get their members under the flat TTree names; see
  // rntupleprojection. Has to be set before the first fill, which is what builds the schema.
  void setFlatProjections(bool flatProjections) { m_flatProjections = flatProjections; }
  void fill(const edm::LuminosityBlockForOutput& iLumi, TFile& file);
  void finalizeWrite();

private:
  void createFields(const edm::LuminosityBlockForOutput& iLumi, TFile& file);
  bool m_flatProjections = true;
  std::vector<edm::EDGetToken> m_counterTableTokens;
  std::vector<edm::EDGetToken> m_flatTableTokens;
  std::unique_ptr<ROOT::RNTupleWriter> m_ntuple;
  RNTupleFieldPtr<std::uint32_t> m_run;
  RNTupleFieldPtr<std::uint32_t> m_luminosityBlock;
  std::vector<SummaryTableOutputFields> m_counterTables;
  TableCollectionSet m_flatTables;
};

class RunNTuple {
public:
  RunNTuple() = default;
  void registerCounterTableToken(const edm::EDGetToken& token);
  void registerFlatTableToken(const edm::EDGetToken& token);
  // Whether the collections also get their members under the flat TTree names; see
  // rntupleprojection. Has to be set before the first fill, which is what builds the schema.
  void setFlatProjections(bool flatProjections) { m_flatProjections = flatProjections; }
  void fill(const edm::RunForOutput& iRun, TFile& file);
  void finalizeWrite();

private:
  void createFields(const edm::RunForOutput& iRun, TFile& file);
  bool m_flatProjections = true;
  std::vector<edm::EDGetToken> m_counterTableTokens;
  std::vector<edm::EDGetToken> m_flatTableTokens;
  std::unique_ptr<ROOT::RNTupleWriter> m_ntuple;
  RNTupleFieldPtr<std::uint32_t> m_run;
  std::vector<SummaryTableOutputFields> m_counterTables;
  TableCollectionSet m_flatTables;
};

// Provenance, in contrast, is written in one pass at the end of the job. These are namespaced
// because a plugin library exports them: the generic EDM RNTuple output in FWIO/RNTupleTemp*
// writes the same two ntuples, and a job loading both plugins would otherwise risk an ODR clash.
namespace rntupleprovenance {
  // Writes the parameter set registry as the ParameterSets ntuple.
  void writeParameterSets(TFile& file);
  // Writes the process history as the MetaData ntuple.
  void writeProcessHistory(const edm::ProcessHistoryRegistry& procHist, TFile& file);
}  // namespace rntupleprovenance

#endif
