#ifndef MuonReco_MuonSelectors_h
#define MuonReco_MuonSelectors_h
//
// Package:    MuonReco
// 
//
// Original Author:  Jake Ribnik, Dmytro Kovalskyi
// $Id: MuonSelectors.h,v 1.2 2008/04/30 22:58:14 dmytro Exp $

#include "DataFormats/MuonReco/interface/Muon.h"
#include "TMath.h"

namespace muon {
   /// main GoodMuon wrapper call
   bool isGoodMuon( const reco::Muon& muon, reco::Muon::SelectionType type );

   // ===========================================================================
   //                               Support functions
   // 
   enum AlgorithmType { TMLastStation, TM2DCompatibility };
   
   // specialized GoodMuon functions called from main wrapper
   bool isGoodMuon( const reco::Muon& muon, 
		    AlgorithmType type,
		    double minCompatibility);
   
   bool isGoodMuon( const reco::Muon& muon, 
		    AlgorithmType type,
		    int minNumberOfMatches,
		    double maxAbsDx,
		    double maxAbsPullX,
		    double maxAbsDy,
		    double maxAbsPullY,
		    double maxChamberDist,
		    double maxChamberDistPull,
		    reco::Muon::ArbitrationType arbitrationType );
   
   // determine if station was crossed well withing active volume
   unsigned int RequiredStationMask( const reco::Muon& muon,
				     double maxChamberDist,
				     double maxChamberDistPull,
				     reco::Muon::ArbitrationType arbitrationType );

   // ------------ method to return the calo compatibility for a track with matched muon info  ------------
   float caloCompatibility(const reco::Muon& muon);

   // ------------ method to calculate the segment compatibility for a track with matched muon info  ------------
   float segmentCompatibility(const reco::Muon& muon);


}
#endif
