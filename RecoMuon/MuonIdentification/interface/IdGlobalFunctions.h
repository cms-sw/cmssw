#ifndef MuonIdentification_IdGlobalFunctions_h
#define MuonIdentification_IdGlobalFunctions_h
//
// Package:    MuonIdentification
// 
//
// Original Author:  Jake Ribnik, Dmytro Kovalskyi
// $Id: IdGlobalFunctions.h,v 1.2 2007/07/26 00:26:35 dmytro Exp $

#include "DataFormats/MuonReco/interface/Muon.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TMath.h"

namespace muonid {
   enum AlgorithmType { TMLastStation };
   enum SelectionType { TMLastStationLoose, TMLastStationTight };
   
   bool isGoodMuon( const reco::Muon& muon, SelectionType type = TMLastStationLoose );
   
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

   // ------------ method to calculate the calo compatibility for a track with matched muon info  ------------
   float getCaloCompatibility(const reco::Muon& muon);

   // ------------ method to calculate the segment compatibility for a track with matched muon info  ------------
   float getSegmentCompatibility(const reco::Muon& muon);


}
#endif
