#ifndef MuonReco_MuonSelectors_h
#define MuonReco_MuonSelectors_h
//
// Package:    MuonReco
// 
//
// Original Author:  Jake Ribnik, Dmytro Kovalskyi
// $Id: MuonSelectors.h,v 1.4 2009/03/27 15:45:23 dmytro Exp $

#include "DataFormats/MuonReco/interface/Muon.h"
#include "TMath.h"
#include <string>

namespace muon {
   /// Selector type
   enum SelectionType {
        All,                      // dummy options - always true
	AllGlobalMuons,           // checks isGlobalMuon flag
	AllStandAloneMuons,       // checks isStandAloneMuon flag
	AllTrackerMuons,          // checks isTrackerMuon flag
	TrackerMuonArbitrated,    // resolve ambiguity of sharing segments
	AllArbitrated,            // all muons with the tracker muon arbitrated
	GlobalMuonPromptTight,    // global muons with tighter fit requirements
	TMLastStationLoose,       // penetration depth loose selector
	TMLastStationTight,       // penetration depth tight selector
	TM2DCompatibilityLoose,   // likelihood based loose selector
	TM2DCompatibilityTight,   // likelihood based tight selector
	TMOneStationLoose,        // require one well matched segment
	TMOneStationTight,        // require one well matched segment
	TMLastStationOptimizedLowPtLoose, // combination of TMLastStation and TMOneStation
	TMLastStationOptimizedLowPtTight  // combination of TMLastStation and TMOneStation
   };

   /// a lightweight "map" for selection type string label and enum value
   struct SelectionTypeStringToEnum { const char *label; SelectionType value; };
   SelectionType selectionTypeFromString( std::string &label );
     
   /// main GoodMuon wrapper call
   bool isGoodMuon( const reco::Muon& muon, SelectionType type );

   // ===========================================================================
   //                               Support functions
   // 
   enum AlgorithmType { TMLastStation, TM2DCompatibility, TMOneStation };
   
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
