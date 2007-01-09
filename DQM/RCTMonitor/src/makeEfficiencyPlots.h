#ifndef makeEfficiencyPlots_h
#define makeEfficiencyPlots_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class makeEfficiencyPlots : public edm::EDAnalyzer {
public:
  explicit makeEfficiencyPlots(const edm::ParameterSet&);
  ~makeEfficiencyPlots();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
private:
  // ----------member data ---------------------------
};

#endif
