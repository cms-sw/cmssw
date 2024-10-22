// -*- C++ -*-
//
// Package:    LHEVpTFilter
// Class:      LHEVpTFilter
//
/*

 Description: Filter to select events with V pT in a given range.
 (Based on LHEGenericFilter)

*/
//

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Math/Vector4D.h"
#include "Math/Vector4Dfwd.h"

#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include <cmath>
#include <cstdlib>
#include <vector>

//
// class declaration
//

class LHEVpTFilter : public edm::global::EDFilter<> {
public:
  explicit LHEVpTFilter(const edm::ParameterSet&);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  const edm::EDGetTokenT<LHEEventProduct> src_;
  const double vptMin_;  // number of particles required to pass filter
  const double vptMax_;  // number of particles required to pass filter
};

LHEVpTFilter::LHEVpTFilter(const edm::ParameterSet& iConfig)
    : src_(consumes<LHEEventProduct>(iConfig.getParameter<edm::InputTag>("src"))),
      vptMin_(iConfig.getParameter<double>("VpTMin")),
      vptMax_(iConfig.getParameter<double>("VpTMax")) {}

// ------------ method called to skim the data  ------------
bool LHEVpTFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  edm::Handle<LHEEventProduct> EvtHandle;
  iEvent.getByToken(src_, EvtHandle);

  std::vector<lhef::HEPEUP::FiveVector> const& lheParticles = EvtHandle->hepeup().PUP;

  std::vector<ROOT::Math::PxPyPzEVector> lepCands;
  for (unsigned int i = 0; i < lheParticles.size(); ++i) {
    if (EvtHandle->hepeup().ISTUP[i] != 1) {  // keep only outgoing particles
      continue;
    }
    unsigned absPdgId = std::abs(EvtHandle->hepeup().IDUP[i]);
    if (absPdgId >= 11 && absPdgId <= 16) {
      lepCands.push_back(
          ROOT::Math::PxPyPzEVector(lheParticles[i][0], lheParticles[i][1], lheParticles[i][2], lheParticles[i][3]));
    }
  }
  double vpt_ = -1;
  if (lepCands.size() == 2) {
    vpt_ = (lepCands[0] + lepCands[1]).pt();
  }
  if (vpt_ < vptMax_ && vpt_ >= vptMin_) {
    return true;
  } else {
    return false;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(LHEVpTFilter);
