// -*- C++ -*-
//
// Package:     EgammaHLTAlgos
// Class  :     EgammaHLTTrackIsolation
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Monica Vazquez Acosta
//         Created:  Tue Jun 13 12:17:19 CEST 2006
//

// system include files

// user include files
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHLTTrackIsolation.h"

std::pair<int, float> EgammaHLTTrackIsolation::electronIsolation(const reco::Track* const tr,
                                                                 const reco::TrackCollection* isoTracks) {
  GlobalPoint vtx(0, 0, tr->vertex().z());
  const reco::Track::Vector& p = tr->momentum();
  GlobalVector mom(p.x(), p.y(), p.z());
  return findIsoTracks(mom, vtx, isoTracks, true);
}

std::pair<int, float> EgammaHLTTrackIsolation::electronIsolation(const reco::Track* const tr,
                                                                 const reco::TrackCollection* isoTracks,
                                                                 GlobalPoint zvtx) {
  // Just to insure consistency with no-vertex-code
  GlobalPoint vtx(0, 0, zvtx.z());
  const reco::Track::Vector& p = tr->momentum();
  GlobalVector mom(p.x(), p.y(), p.z());
  return findIsoTracks(mom, vtx, isoTracks, true);
}

std::pair<int, float> EgammaHLTTrackIsolation::electronIsolation(const reco::Track* const tr,
                                                                 const reco::ElectronCollection* allEle,
                                                                 const reco::TrackCollection* isoTracks) {
  GlobalPoint vtx(0, 0, tr->vertex().z());
  const reco::Track::Vector& p = tr->momentum();
  GlobalVector mom(p.x(), p.y(), p.z());
  return findIsoTracksWithoutEle(mom, vtx, allEle, isoTracks);
}

std::pair<int, float> EgammaHLTTrackIsolation::photonIsolation(const reco::RecoCandidate* const recocandidate,
                                                               const reco::TrackCollection* isoTracks,
                                                               bool useVertex) {
  if (useVertex) {
    GlobalPoint vtx(0, 0, recocandidate->vertex().z());
    return photonIsolation(recocandidate, isoTracks, vtx);
  } else {
    reco::RecoCandidate::Point pos = recocandidate->superCluster()->position();
    GlobalVector mom(pos.x(), pos.y(), pos.z());
    return findIsoTracks(mom, GlobalPoint(), isoTracks, false, false);
  }
}

std::pair<int, float> EgammaHLTTrackIsolation::photonIsolation(const reco::RecoCandidate* const recocandidate,
                                                               const reco::TrackCollection* isoTracks,
                                                               GlobalPoint zvtx) {
  // to insure consistency with no-free-vertex-code
  GlobalPoint vtx(0, 0, zvtx.z());

  reco::RecoCandidate::Point pos = recocandidate->superCluster()->position();
  GlobalVector mom(pos.x() - vtx.x(), pos.y() - vtx.y(), pos.z() - vtx.z());

  return findIsoTracks(mom, vtx, isoTracks, false);
}

std::pair<int, float> EgammaHLTTrackIsolation::photonIsolation(const reco::RecoCandidate* const recocandidate,
                                                               const reco::ElectronCollection* allEle,
                                                               const reco::TrackCollection* isoTracks) {
  reco::RecoCandidate::Point pos = recocandidate->superCluster()->position();
  GlobalVector mom(pos.x(), pos.y(), pos.z());
  return findIsoTracksWithoutEle(mom, GlobalPoint(), allEle, isoTracks);
}

std::pair<int, float> EgammaHLTTrackIsolation::findIsoTracks(
    GlobalVector mom, GlobalPoint vtx, const reco::TrackCollection* isoTracks, bool isElectron, bool useVertex) {
  // Check that reconstructed tracks fit within cone boundaries,
  // (Note: tracks will not always stay within boundaries)
  int ntrack = 0;
  float ptSum = 0.;

  for (reco::TrackCollection::const_iterator trItr = isoTracks->begin(); trItr != isoTracks->end(); ++trItr) {
    GlobalPoint ivtx(trItr->vertex().x(), trItr->vertex().y(), trItr->vertex().z());
    reco::Track::Vector ip = trItr->momentum();
    GlobalVector imom(ip.x(), ip.y(), ip.z());

    float pt = imom.perp();
    float dperp = 0.;
    float dz = 0.;
    float deta = 0.;
    float dphi = 0.;
    if (useVertex) {
      dperp = ivtx.perp() - vtx.perp();
      dz = ivtx.z() - vtx.z();
      deta = imom.eta() - mom.eta();
      dphi = imom.phi() - mom.phi();
    } else {
      //in case of unkown photon vertex, modify direction of photon to point from
      //current track vertex to sc instead of from (0.,0.,0.) to sc.  In this
      //way, all tracks are considered based on direction alone.
      GlobalVector mom_temp = mom - GlobalVector(ivtx.x(), ivtx.y(), ivtx.z());
      deta = imom.eta() - mom_temp.eta();
      dphi = imom.phi() - mom_temp.phi();
    }
    // Correct dmom_phi's from [-2pi,2pi] to [-pi,pi]
    if (dphi > M_PI)
      dphi = dphi - 2 * M_PI;
    else if (dphi < -M_PI)
      dphi = dphi + 2 * M_PI;

    float R = sqrt(dphi * dphi + deta * deta);

    // Apply boundary cut
    // bool selected=false;

    // if (pt > ptMin && R < conesize &&
    //	fabs(dperp) < rspan && fabs(dz) < zspan) selected=true;

    // if (selected) {
    //  ntrack++;
    //  if (!isElectron || R > vetoConesize) ptSum+=pt; //to exclude electron track
    // }
    // float theVetoVar = R;
    // if (isElectron) theVetoVar = R;

    //hmm how do I figure out if this is barrel or endcap?
    //abs(mom.eta())<1.5 is the obvious choice but that will be electron not detector eta for electrons
    //well lets leave it as that for now, its what reco does (well with eta=1.479)
    double innerStrip = fabs(mom.eta()) < 1.479 ? stripBarrel : stripEndcap;

    if (pt > ptMin && R < conesize && R > vetoConesize && fabs(dperp) < rspan && fabs(dz) < zspan &&
        fabs(deta) >= innerStrip) {
      ntrack++;
      ptSum += pt;
    }
  }

  // if (isElectron) ntrack-=1; //to exclude electron track

  return (std::pair<int, float>(ntrack, ptSum));
}

std::pair<int, float> EgammaHLTTrackIsolation::findIsoTracksWithoutEle(GlobalVector mom,
                                                                       GlobalPoint vtx,
                                                                       const reco::ElectronCollection* allEle,
                                                                       const reco::TrackCollection* isoTracks) {
  // Check that reconstructed tracks fit within cone boundaries,
  // (Note: tracks will not always stay within boundaries)
  int ntrack = 0;
  float ptSum = 0.;
  std::vector<float> etaele;
  std::vector<float> phiele;

  // std::cout << "allEle.size() = " << allEle->size() << std::endl;

  // Store ALL electrons eta and phi
  for (reco::ElectronCollection::const_iterator iElectron = allEle->begin(); iElectron != allEle->end(); iElectron++) {
    reco::TrackRef anothereletrackref = iElectron->track();
    etaele.push_back(anothereletrackref->momentum().eta());
    phiele.push_back(anothereletrackref->momentum().phi());
  }

  for (reco::TrackCollection::const_iterator trItr = isoTracks->begin(); trItr != isoTracks->end(); ++trItr) {
    GlobalPoint ivtx(trItr->vertex().x(), trItr->vertex().y(), trItr->vertex().z());
    reco::Track::Vector ip = trItr->momentum();
    GlobalVector imom(ip.x(), ip.y(), ip.z());

    float pt = imom.perp();
    float dperp = ivtx.perp() - vtx.perp();
    float dz = ivtx.z() - vtx.z();
    float deta = imom.eta() - mom.eta();
    float dphi = imom.phi() - mom.phi();

    // Correct dmom_phi's from [-2pi,2pi] to [-pi,pi]
    if (dphi > M_PI)
      dphi = dphi - 2 * M_PI;
    else if (dphi < -M_PI)
      dphi = dphi + 2 * M_PI;

    float R = sqrt(dphi * dphi + deta * deta);

    // Apply boundary cut
    bool selected = false;
    bool passedconeveto = true;

    //hmm how do I figure out if this is barrel or endcap?
    //abs(mom.eta())<1.5 is the obvious choice but that will be electron not detector eta for electrons
    //well lets leave it as that for now, its what reco does (well with eta=1.479)
    double innerStrip = fabs(mom.eta()) < 1.479 ? stripBarrel : stripEndcap;

    if (pt > ptMin && R < conesize && fabs(dperp) < rspan && fabs(dz) < zspan && fabs(deta) >= innerStrip)
      selected = true;

    // Check that NO electron is counted in the isolation
    for (unsigned int eleItr = 0; eleItr < etaele.size(); ++eleItr) {
      deta = etaele[eleItr] - imom.eta();
      dphi = phiele[eleItr] - imom.phi();

      // Correct dmom_phi's from [-2pi,2pi] to [-pi,pi]
      if (dphi > M_PI)
        dphi = dphi - 2 * M_PI;
      else if (dphi < -M_PI)
        dphi = dphi + 2 * M_PI;

      R = sqrt(dphi * dphi + deta * deta);
      if (R < vetoConesize)
        passedconeveto = false;
    }

    if (selected && passedconeveto) {
      ntrack++;
      ptSum += pt;  //to exclude electron tracks
    }
  }

  // ntrack-=1; //to exclude electron track

  return (std::pair<int, float>(ntrack, ptSum));
}
