#ifndef MuonIdentification_MuonArbitrationMethods_h
#define MuonIdentification_MuonArbitrationMethods_h

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

// Author: Jake Ribnik (UCSB)

/// functor predicate for standard library sort algorithm
struct SortMuonSegmentMatches {
   /// constructor takes arbitration type
   SortMuonSegmentMatches( unsigned int flag ) {
      flag_ = flag;
   }
   /// sorts vector of pairs of chamber and segment pointers
   bool operator() ( std::pair<reco::MuonChamberMatch*,reco::MuonSegmentMatch*> p1,
         std::pair<reco::MuonChamberMatch*,reco::MuonSegmentMatch*> p2 )
   {
      reco::MuonChamberMatch* cm1 = p1.first;
      reco::MuonSegmentMatch* sm1 = p1.second;
      reco::MuonChamberMatch* cm2 = p2.first;
      reco::MuonSegmentMatch* sm2 = p2.second;

      if(flag_ == reco::MuonSegmentMatch::BestInChamberByDX ||
         flag_ == reco::MuonSegmentMatch::BestInStationByDX ||
         flag_ == reco::MuonSegmentMatch::BelongsToTrackByDX)
         return fabs(sm1->x-cm1->x) < fabs(sm2->x-cm2->x);
      if(flag_ == reco::MuonSegmentMatch::BestInChamberByDR ||
         flag_ == reco::MuonSegmentMatch::BestInStationByDR || 
         flag_ == reco::MuonSegmentMatch::BelongsToTrackByDR)
      {
         if((! sm1->hasZed()) || (! sm2->hasZed())) // no y information so return dx
            return fabs(sm1->x-cm1->x) < fabs(sm2->x-cm2->x);
         return sqrt(pow(sm1->x-cm1->x,2)+pow(sm1->y-cm1->y,2)) <
            sqrt(pow(sm2->x-cm2->x,2)+pow(sm2->y-cm2->y,2)); 
      }
      if(flag_ == reco::MuonSegmentMatch::BestInChamberByDXSlope ||
         flag_ == reco::MuonSegmentMatch::BestInStationByDXSlope ||
         flag_ == reco::MuonSegmentMatch::BelongsToTrackByDXSlope)
         return fabs(sm1->dXdZ-cm1->dXdZ) < fabs(sm2->dXdZ-cm2->dXdZ);
      if(flag_ == reco::MuonSegmentMatch::BestInChamberByDRSlope ||
         flag_ == reco::MuonSegmentMatch::BestInStationByDRSlope ||
         flag_ == reco::MuonSegmentMatch::BelongsToTrackByDRSlope)
      {
         if((! sm1->hasZed()) || (! sm2->hasZed())) // no y information so return dx
            return fabs(sm1->dXdZ-cm1->dXdZ) < fabs(sm2->dXdZ-cm2->dXdZ);
         return sqrt(pow(sm1->dXdZ-cm1->dXdZ,2)+pow(sm1->dYdZ-cm1->dYdZ,2)) <
            sqrt(pow(sm2->dXdZ-cm2->dXdZ,2)+pow(sm2->dYdZ-cm2->dYdZ,2)); 
      }

      return false; // is this appropriate? fix this
   }

   unsigned int flag_;
};

#endif
