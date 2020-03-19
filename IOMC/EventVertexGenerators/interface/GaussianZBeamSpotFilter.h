#ifndef GaussianZBeamSpotFilter_h
#define GaussianZBeamSpotFilter_h

// Filter to select events with a gaussian Z beam spot shape
// narrower than the original one

// system include files

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class GaussianZBeamSpotFilter : public edm::EDFilter {
public:
  explicit GaussianZBeamSpotFilter(const edm::ParameterSet&);
  ~GaussianZBeamSpotFilter() override;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  edm::InputTag src_;
  double baseSZ_;
  double baseZ0_;
  double newSZ_;
  double newZ0_;
};

#endif
