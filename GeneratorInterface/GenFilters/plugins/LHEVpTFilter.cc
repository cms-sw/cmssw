#include "GeneratorInterface/GenFilters/plugins/LHEVpTFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <cmath>
#include <vector>

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
  if (vpt_ <= vptMax_ && vpt_ > vptMin_) {
    return true;
  } else {
    return false;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(LHEVpTFilter);
