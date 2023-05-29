#ifndef GaussianZBeamSpotFilter_h
#define GaussianZBeamSpotFilter_h

// Filter to select events with a gaussian Z beam spot shape
// narrower than the original one

// system include files

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

class GaussianZBeamSpotFilter : public edm::stream::EDFilter<> {
public:
  explicit GaussianZBeamSpotFilter(const edm::ParameterSet&);
  ~GaussianZBeamSpotFilter() override = default;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  const edm::InputTag src_;
  const double baseSZ_;
  const double baseZ0_;
  const double newSZ_;
  const double newZ0_;
  const edm::EDGetTokenT<edm::HepMCProduct> srcToken_;
};

#endif
