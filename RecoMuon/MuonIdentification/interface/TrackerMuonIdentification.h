#ifndef MuonIdentification_TrackerMuonIdentification_h
#define MuonIdentification_TrackerMuonIdentification_h
//
// Package:    MuonIdentification
// Class:      TrackerMuonIdentification
// 
//
// Original Author:  Jake Ribnik, Dmytro Kovalskyi
// $Id$

#include "DataFormats/MuonReco/interface/Muon.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TrackerMuonIdentification
{
 public:
   enum AlgorithmType { LastStation };
   void setParameters(const edm::ParameterSet&);
   bool isGoodMuon( const reco::Muon& ) const;

 private:
   // determine if station was crossed well withing active volume
   unsigned int RequiredStationMask( const reco::Muon& ) const;
   AlgorithmType type_;
   double minPt_;
   double minP_;
   int minNumberOfMatches_;
   double maxAbsDx_;
   double maxAbsPullX_;
   double maxAbsDy_;
   double maxAbsPullY_;
   double maxChamberDist_;
   reco::Muon::ArbitrationType arbitrationType_;
};
#endif
