#ifndef MuonReco_MuonSelectors_h
#define MuonReco_MuonSelectors_h
//
// Package:    MuonReco
// 
//
// Original Author:  Jake Ribnik, Dmytro Kovalskyi
// $Id: MuonSelectors.h,v 1.7 2009/09/09 20:55:46 aeverett Exp $

#include "DataFormats/MuonReco/interface/Muon.h"
#include "TMath.h"
#include <string>

namespace muon {
   /// Selector type
   enum SelectionType {
      All = 0,                      // dummy options - always true
      AllGlobalMuons = 1,           // checks isGlobalMuon flag
      AllStandAloneMuons = 2,       // checks isStandAloneMuon flag
      AllTrackerMuons = 3,          // checks isTrackerMuon flag
      TrackerMuonArbitrated = 4,    // resolve ambiguity of sharing segments
      AllArbitrated = 5,            // all muons with the tracker muon arbitrated
      GlobalMuonPromptTight = 6,    // global muons with tighter fit requirements
      TMLastStationLoose = 7,       // penetration depth loose selector
      TMLastStationTight = 8,       // penetration depth tight selector
      TM2DCompatibilityLoose = 9,   // likelihood based loose selector
      TM2DCompatibilityTight = 10,  // likelihood based tight selector
      TMOneStationLoose = 11,       // require one well matched segment
      TMOneStationTight = 12,       // require one well matched segment
      TMLastStationOptimizedLowPtLoose = 13, // combination of TMLastStation and TMOneStation
      TMLastStationOptimizedLowPtTight = 14, // combination of TMLastStation and TMOneStation
      GMTkChiCompatibility = 15,    // require tk stub have good chi2 relative to glb track
      GMStaChiCompatibility = 16,   // require sta stub have good chi2 compatibility relative to glb track
      GMTkKinkTight = 17,           // require a small kink value in the tracker stub
      TMLastStationAngLoose = 18,   // TMLastStationLoose with additional angular cuts
      TMLastStationAngTight = 19,   // TMLastStationTight with additional angular cuts
      TMOneStationAngLoose = 20,    // TMOneStationLoose with additional angular cuts
      TMOneStationAngTight = 21     // TMOneStationTight with additional angular cuts
   };

   /// a lightweight "map" for selection type string label and enum value
   struct SelectionTypeStringToEnum { const char *label; SelectionType value; };
   SelectionType selectionTypeFromString( const std::string &label );
     
   /// main GoodMuon wrapper call
   bool isGoodMuon( const reco::Muon& muon, SelectionType type );

   // ===========================================================================
   //                               Support functions
   // 
   enum AlgorithmType { TMLastStation, TM2DCompatibility, TMOneStation };
   
   // specialized GoodMuon functions called from main wrapper
   bool isGoodMuon( const reco::Muon& muon, 
		    AlgorithmType type,
		    double minCompatibility,
		    reco::Muon::ArbitrationType arbitrationType );
   
   bool isGoodMuon( const reco::Muon& muon, 
		    AlgorithmType type,
		    int minNumberOfMatches,
		    double maxAbsDx,
		    double maxAbsPullX,
		    double maxAbsDy,
		    double maxAbsPullY,
		    double maxChamberDist,
		    double maxChamberDistPull,
		    reco::Muon::ArbitrationType arbitrationType,
            bool   syncMinNMatchesNRequiredStationsInBarrelOnly,
            bool   applyAlsoAngularCuts);
   
   // determine if station was crossed well withing active volume
   unsigned int RequiredStationMask( const reco::Muon& muon,
				     double maxChamberDist,
				     double maxChamberDistPull,
				     reco::Muon::ArbitrationType arbitrationType );

   // ------------ method to return the calo compatibility for a track with matched muon info  ------------
   float caloCompatibility(const reco::Muon& muon);

   // ------------ method to calculate the segment compatibility for a track with matched muon info  ------------
   float segmentCompatibility(const reco::Muon& muon,reco::Muon::ArbitrationType arbitrationType = reco::Muon::SegmentAndTrackArbitration);


}
#endif
