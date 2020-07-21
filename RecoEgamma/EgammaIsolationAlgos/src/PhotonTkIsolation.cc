//*****************************************************************************
// File:      PhotonTkIsolation.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

#include "RecoEgamma/EgammaIsolationAlgos/interface/PhotonTkIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTrackSelector.h"

PhotonTkIsolation::PhotonTkIsolation(float extRadius,
                                     float intRadiusBarrel,
                                     float intRadiusEndcap,
                                     float stripBarrel,
                                     float stripEndcap,
                                     float etLow,
                                     float lip,
                                     float drb,
                                     const reco::TrackCollection* trackCollection,
                                     reco::TrackBase::Point beamPoint,
                                     const std::string& dzOptionString)
    : extRadius2_(extRadius * extRadius),
      intRadiusBarrel2_(intRadiusBarrel * intRadiusBarrel),
      intRadiusEndcap2_(intRadiusEndcap * intRadiusEndcap),
      stripBarrel_(stripBarrel),
      stripEndcap_(stripEndcap),
      etLow_(etLow),
      lip_(lip),
      drb_(drb),
      trackCollection_(trackCollection),
      beamPoint_(beamPoint) {
  setDzOption(dzOptionString);
}

void PhotonTkIsolation::setDzOption(const std::string& s) {
  if (!s.compare("dz"))
    dzOption_ = egammaisolation::EgammaTrackSelector::dz;
  else if (!s.compare("vz"))
    dzOption_ = egammaisolation::EgammaTrackSelector::vz;
  else if (!s.compare("bs"))
    dzOption_ = egammaisolation::EgammaTrackSelector::bs;
  else if (!s.compare("vtx"))
    dzOption_ = egammaisolation::EgammaTrackSelector::vtx;
  else
    dzOption_ = egammaisolation::EgammaTrackSelector::dz;
}

PhotonTkIsolation::~PhotonTkIsolation() {}

// unified acces to isolations
std::pair<int, float> PhotonTkIsolation::getIso(const reco::Candidate* photon) const {
  int counter = 0;
  float ptSum = 0.;

  //Take the photon position
  float photonEta = photon->eta();

  //loop over tracks
  for (reco::TrackCollection::const_iterator trItr = trackCollection_->begin(); trItr != trackCollection_->end();
       ++trItr) {
    //check z-distance of vertex
    float dzCut = 0;
    switch (dzOption_) {
      case egammaisolation::EgammaTrackSelector::dz:
        dzCut = fabs((*trItr).dz() - photon->vertex().z());
        break;
      case egammaisolation::EgammaTrackSelector::vz:
        dzCut = fabs((*trItr).vz() - photon->vertex().z());
        break;
      case egammaisolation::EgammaTrackSelector::bs:
        dzCut = fabs((*trItr).dz(beamPoint_) - photon->vertex().z());
        break;
      case egammaisolation::EgammaTrackSelector::vtx:
        dzCut = fabs((*trItr).dz(photon->vertex()));
        break;
      default:
        dzCut = fabs((*trItr).vz() - photon->vertex().z());
        break;
    }
    if (dzCut > lip_)
      continue;

    float this_pt = (*trItr).pt();
    if (this_pt < etLow_)
      continue;
    if (fabs((*trItr).dxy(beamPoint_)) > drb_)
      continue;  // only consider tracks from the main vertex
    float dr2 = reco::deltaR2(*trItr, *photon);
    float deta = (*trItr).eta() - photonEta;
    if (fabs(photonEta) < 1.479) {
      if (dr2 < extRadius2_ && dr2 >= intRadiusBarrel2_ && fabs(deta) >= stripBarrel_) {
        ++counter;
        ptSum += this_pt;
      }
    } else {
      if (dr2 < extRadius2_ && dr2 >= intRadiusEndcap2_ && fabs(deta) >= stripEndcap_) {
        ++counter;
        ptSum += this_pt;
      }
    }

  }  //end loop over tracks

  std::pair<int, float> retval;
  retval.first = counter;
  retval.second = ptSum;
  return retval;
}
