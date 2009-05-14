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
// $Id: EgammaHLTTrackIsolation.cc,v 1.1 2006/06/20 11:28:33 monicava Exp $
//

// system include files

// user include files
#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTTrackIsolation.h"



std::pair<int,float> EgammaHLTTrackIsolation::electronIsolation(const reco::Track * const tr, const reco::TrackCollection* isoTracks)
{
  GlobalPoint vtx(0,0,tr->vertex().z());
  reco::Track::Vector p = tr->momentum();
  GlobalVector mom( p.x(), p.y(), p.z() );
  return findIsoTracks(mom,vtx,isoTracks,true);
}


std::pair<int,float> EgammaHLTTrackIsolation::electronIsolation(const reco::Track *  const tr, const reco::TrackCollection* isoTracks, GlobalPoint zvtx)
{ 
  // Just to insure consistency with no-vertex-code
  GlobalPoint vtx(0,0,zvtx.z());
  reco::Track::Vector p = tr->momentum();
  GlobalVector mom( p.x(), p.y(), p.z() );
  return findIsoTracks(mom,vtx,isoTracks,true);
}


std::pair<int,float> EgammaHLTTrackIsolation::photonIsolation(const reco::RecoCandidate * const recocandidate, const reco::TrackCollection* isoTracks, bool useVertex)
{

  if (useVertex) {
    GlobalPoint vtx(0,0,recocandidate->vertex().z());
    return photonIsolation(recocandidate,isoTracks,vtx);
  } else {
    reco::RecoCandidate::Point pos = recocandidate->superCluster()->position();
    GlobalVector mom(pos.x(),pos.y(),pos.z());
    return findIsoTracks(mom,GlobalPoint(),isoTracks,false,false);
  }

}

std::pair<int,float> EgammaHLTTrackIsolation::photonIsolation(const reco::RecoCandidate * const recocandidate, const reco::TrackCollection* isoTracks, GlobalPoint zvtx)
{

  // to insure consistency with no-free-vertex-code
  GlobalPoint vtx(0,0,zvtx.z());

  reco::RecoCandidate::Point pos = recocandidate->superCluster()->position();
  GlobalVector mom(pos.x()-vtx.x(),pos.y()-vtx.y(),pos.z()-vtx.z());

  return findIsoTracks(mom,vtx,isoTracks,false);

}


std::pair<int,float> EgammaHLTTrackIsolation::findIsoTracks(GlobalVector mom, GlobalPoint vtx,  const reco::TrackCollection* isoTracks, bool isElectron, bool useVertex)
{

  // Check that reconstructed tracks fit within cone boundaries,
  // (Note: tracks will not always stay within boundaries)
  int ntrack = 0;
  float ptSum = 0.;

  for(reco::TrackCollection::const_iterator trItr = isoTracks->begin(); trItr != isoTracks->end(); ++trItr){

    GlobalPoint ivtx(trItr->vertex().x(),trItr->vertex().y(),trItr->vertex().z());
    reco::Track::Vector ip = trItr->momentum();
    GlobalVector imom ( ip.x(), ip.y(), ip.z());

    float pt = imom.perp();
    float dperp = 0.;
    float dz = 0.;
    float deta = 0.;
    float dphi = 0.;
    if (useVertex) {
      dperp = ivtx.perp()-vtx.perp();
      dz = ivtx.z()-vtx.z();
      deta = imom.eta()-mom.eta();
      dphi = imom.phi()-mom.phi();
    } else {
      //in case of unkown photon vertex, modify direction of photon to point from
      //current track vertex to sc instead of from (0.,0.,0.) to sc.  In this 
      //way, all tracks are considered based on direction alone.
      GlobalVector mom_temp = mom - GlobalVector(ivtx.x(),ivtx.y(),ivtx.z());
      deta = imom.eta()-mom_temp.eta();
      dphi = imom.phi()-mom_temp.phi();
    }
    // Correct dmom_phi's from [-2pi,2pi] to [-pi,pi]
    if (dphi>M_PI) dphi = dphi - 2*M_PI;
    else if (dphi<-M_PI) dphi = dphi + 2*M_PI;

    float R = sqrt( dphi*dphi + deta*deta );

    // Apply boundary cut
    bool selected=false;

    if (pt > ptMin && R < conesize &&
	fabs(dperp) < rspan && fabs(dz) < zspan) selected=true;
  
    if (selected) {
      ntrack++;
      if (!isElectron || R > vetoConesize) ptSum+=pt; //to exclude electron track
    }

  }

  if (isElectron) ntrack-=1; //to exclude electron track

  return (std::pair<int,float>(ntrack,ptSum));

}
