//*****************************************************************************
// File:      ElectronTkIsolation.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

#include "RecoEgamma/EgammaIsolationAlgos/interface/ElectronTkIsolation.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include <Math/VectorUtil.h>

ElectronTkIsolation::ElectronTkIsolation(double extRadius,
                                         double intRadiusBarrel,
                                         double intRadiusEndcap,
                                         double stripBarrel,
                                         double stripEndcap,
                                         double ptLow,
                                         double lip,
                                         double drb,
                                         const reco::TrackCollection* trackCollection,
                                         reco::TrackBase::Point beamPoint,
                                         const std::string& dzOptionString)
    : extRadius_(extRadius),
      intRadiusBarrel_(intRadiusBarrel),
      intRadiusEndcap_(intRadiusEndcap),
      stripBarrel_(stripBarrel),
      stripEndcap_(stripEndcap),
      ptLow_(ptLow),
      lip_(lip),
      drb_(drb),
      trackCollection_(trackCollection),
      beamPoint_(beamPoint) {
  setAlgosToReject();
  setDzOption(dzOptionString);
}

ElectronTkIsolation::~ElectronTkIsolation() {}

std::pair<int, double> ElectronTkIsolation::getIso(const reco::GsfElectron* electron) const {
  return getIso(&(*(electron->gsfTrack())));
}

// unified acces to isolations
std::pair<int, double> ElectronTkIsolation::getIso(const reco::Track* tmpTrack) const {
  int counter = 0;
  double ptSum = 0.;
  //Take the electron track
  math::XYZVector tmpElectronMomentumAtVtx = (*tmpTrack).momentum();
  double tmpElectronEtaAtVertex = (*tmpTrack).eta();

  for (reco::TrackCollection::const_iterator itrTr = (*trackCollection_).begin(); itrTr != (*trackCollection_).end();
       ++itrTr) {
    double this_pt = (*itrTr).pt();
    if (this_pt < ptLow_)
      continue;

    double dzCut = 0;
    switch (dzOption_) {
      case egammaisolation::EgammaTrackSelector::dz:
        dzCut = fabs((*itrTr).dz() - (*tmpTrack).dz());
        break;
      case egammaisolation::EgammaTrackSelector::vz:
        dzCut = fabs((*itrTr).vz() - (*tmpTrack).vz());
        break;
      case egammaisolation::EgammaTrackSelector::bs:
        dzCut = fabs((*itrTr).dz(beamPoint_) - (*tmpTrack).dz(beamPoint_));
        break;
      case egammaisolation::EgammaTrackSelector::vtx:
        dzCut = fabs((*itrTr).dz(tmpTrack->vertex()));
        break;
      default:
        dzCut = fabs((*itrTr).vz() - (*tmpTrack).vz());
        break;
    }
    if (dzCut > lip_)
      continue;
    if (fabs((*itrTr).dxy(beamPoint_)) > drb_)
      continue;
    double dr = ROOT::Math::VectorUtil::DeltaR(itrTr->momentum(), tmpElectronMomentumAtVtx);
    double deta = (*itrTr).eta() - tmpElectronEtaAtVertex;
    bool isBarrel = std::abs(tmpElectronEtaAtVertex) < 1.479;
    double intRadius = isBarrel ? intRadiusBarrel_ : intRadiusEndcap_;
    double strip = isBarrel ? stripBarrel_ : stripEndcap_;
    if (dr < extRadius_ && dr >= intRadius && std::abs(deta) >= strip && passAlgo(*itrTr)) {
      ++counter;
      ptSum += this_pt;
    }

  }  //end loop over tracks

  std::pair<int, double> retval;
  retval.first = counter;
  retval.second = ptSum;

  return retval;
}

int ElectronTkIsolation::getNumberTracks(const reco::GsfElectron* electron) const {
  //counter for the tracks in the isolation cone
  return getIso(electron).first;
}

double ElectronTkIsolation::getPtTracks(const reco::GsfElectron* electron) const { return getIso(electron).second; }

bool ElectronTkIsolation::passAlgo(const reco::TrackBase& trk) const {
  int algo = trk.algo();
  bool rejAlgo = std::binary_search(algosToReject_.begin(), algosToReject_.end(), algo);
  return rejAlgo == false;
}

void ElectronTkIsolation::setAlgosToReject() {
  algosToReject_ = {reco::TrackBase::jetCoreRegionalStep};
  std::sort(algosToReject_.begin(), algosToReject_.end());
}
