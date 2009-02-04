#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"

namespace susybsm {

float HSCParticle::p() const {
  if(hasMuonCombinedTrack()) return combinedTrack().p();
  if(hasMuonStaTrack()) return staTrack().p();
  if(hasMuonTrack()) return muonTrack().p();
  if(hasMuonTrack()) return trackerTrack().p();
  return 0.;
}

float HSCParticle::pt() const {
  if(hasMuonCombinedTrack()) return combinedTrack().pt();
  if(hasMuonStaTrack()) return staTrack().pt();
  if(hasMuonTrack()) return muonTrack().pt();
  if(hasMuonTrack()) return trackerTrack().pt();
  return 0.;
}

}
