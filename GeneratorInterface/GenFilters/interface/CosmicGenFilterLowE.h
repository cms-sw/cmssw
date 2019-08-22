// daniele.benedetti@cern.ch, livio.fano@cern.ch
#ifndef COSMICGENFILTERLOWE_H
#define COSMICGENFILTERLOWE_H

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TRandom2.h"
#include "TMath.h"

class CosmicGenFilterLowE : public edm::EDFilter {
public:
  explicit CosmicGenFilterLowE(const edm::ParameterSet& conf);
  ~CosmicGenFilterLowE() override {}
  //virtual bool filter(edm::Event & e, edm::EventSetup const& c);
  bool filter(edm::Event& iEvent, edm::EventSetup const& c) override;

private:
  TRandom2 RanGen2;
};

#endif
