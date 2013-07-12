#ifndef MuonIdentification_MuonCosmicsId_h
#define MuonIdentification_MuonCosmicsId_h 1
// -*- C++ -*-
//
// Description: 
//     Tools to identify cosmics muons in collisions
//  
// Original Author:  Dmytro Kovalskyi
// $Id: MuonCosmicsId.h,v 1.1 2010/06/22 13:19:56 dmytro Exp $
//
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/Handle.h"
namespace muonid
{
  // returns angle and dPt/Pt
  std::pair<double, double> matchTracks(const reco::Track& ref,
					const reco::Track& probe);

  reco::TrackRef findOppositeTrack(const edm::Handle<reco::TrackCollection>& collection, 
				   const reco::Track& muon,
				   double angleMatch = 0.01,
				   double momentumMatch = 0.05);
}
#endif
