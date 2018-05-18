#ifndef L1MuonOverlapParamsDBProducer_H
#define L1MuonOverlapParamsDBProducer_H

#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"

class L1TMuonOverlapParams;

class L1MuonOverlapParamsDBProducer : public edm::EDAnalyzer {
public:
  L1MuonOverlapParamsDBProducer(const edm::ParameterSet & cfg);
  ~L1MuonOverlapParamsDBProducer() override{}
  void beginJob() override{};
  void beginRun(const edm::Run&,  const edm::EventSetup& es) override;
  void analyze(const edm::Event&, const edm::EventSetup& es) override;
  void endJob() override{};

private:

  std::unique_ptr<L1TMuonOverlapParams> omtfParams, omtfPatterns;

}; 

#endif
