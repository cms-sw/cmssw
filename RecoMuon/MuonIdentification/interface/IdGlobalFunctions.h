#ifndef MuonIdentification_IdGlobalFunctions_h
#define MuonIdentification_IdGlobalFunctions_h
//
// Package:    MuonIdentification
// 
//
// Original Author:  Jake Ribnik, Dmytro Kovalskyi
// $Id$

#include "DataFormats/MuonReco/interface/Muon.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace muonid {
   enum AlgorithmType { LastStation };
   enum SelectionType { LastStationLoose, LastStationTight };
   
   bool isGoodMuon( const reco::Muon& muon, SelectionType type = LastStationLoose );
   
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

}
#endif
