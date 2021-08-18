#ifndef L1T_OmtfP1_L1MuonOverlapParamsDBProducer_H
#define L1T_OmtfP1_L1MuonOverlapParamsDBProducer_H

#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"

class L1TMuonOverlapParams;

class L1MuonOverlapPhase1ParamsDBProducer : public edm::EDAnalyzer {
public:
  L1MuonOverlapPhase1ParamsDBProducer(const edm::ParameterSet& cfg);
  ~L1MuonOverlapPhase1ParamsDBProducer() override {}
  void beginJob() override{};
  void beginRun(const edm::Run&, const edm::EventSetup& es) override;
  void analyze(const edm::Event&, const edm::EventSetup& es) override;
  void endJob() override{};

private:
  edm::ESGetToken<L1TMuonOverlapParams, L1TMuonOverlapParamsRcd> omtfParamsEsToken;

  std::unique_ptr<L1TMuonOverlapParams> omtfParams, omtfPatterns;
};

#endif
