#include "RecoMuon/MuonIdentification/interface/IdGlobalFunctions.h"

unsigned int muonid::RequiredStationMask( const reco::Muon& muon,
					  double maxChamberDist,
					  double maxChamberDistPull,
					  reco::Muon::ArbitrationType arbitrationType )
{
   unsigned int theMask = 0;

   for(int stationIdx = 1; stationIdx < 5; ++stationIdx)
      for(int detectorIdx = 1; detectorIdx < 3; ++detectorIdx)
         if(muon.trackDist(stationIdx,detectorIdx,arbitrationType) < maxChamberDist &&
               muon.trackDist(stationIdx,detectorIdx,arbitrationType)/muon.trackDistErr(stationIdx,detectorIdx,arbitrationType) < maxChamberDistPull)
            theMask += 1<<((stationIdx-1)+4*(detectorIdx-1));

   return theMask;
}

bool muonid::isGoodMuon( const reco::Muon& muon,
			 AlgorithmType type,
			 int minNumberOfMatches,
			 double maxAbsDx,
			 double maxAbsPullX,
			 double maxAbsDy,
			 double maxAbsPullY,
			 double maxChamberDist,
			 double maxChamberDistPull,
			 reco::Muon::ArbitrationType arbitrationType )
{
   if (!muon.isMatchesValid()) return false;
   bool goodMuon = false;

   unsigned int theStationMask = muon.stationMask(arbitrationType);
   unsigned int theRequiredStationMask = RequiredStationMask(muon, maxChamberDist, maxChamberDistPull, arbitrationType);

   // If there are no required stations, require there be at least two segments
   int numSegs = 0;
   for(int it = 0; it < 8; ++it)
      if(theStationMask & 1<<it) ++numSegs;

   if(numSegs > 1) goodMuon = 1;

   // Require that last required station have segment
   if(theRequiredStationMask)
      for(int stationIdx = 7; stationIdx >= 0; --stationIdx)
         if(theRequiredStationMask & 1<<stationIdx)
            if(theStationMask & 1<<stationIdx) {
               goodMuon &= 1;
               break;
            } else {
               goodMuon = false;
               break;
            }

   if(!goodMuon) return false;

   // Impose pull cuts on last segment
   int lastSegBit = 0;
   for(int stationIdx = 7; stationIdx >= 0; --stationIdx)
      if(theStationMask & 1<<stationIdx) {
         lastSegBit = stationIdx;
         break;
      }

   int station = 0, detector = 0;
   station  = lastSegBit < 4 ? lastSegBit+1 : lastSegBit-3;
   detector = lastSegBit < 4 ? 1 : 2;

   if(lastSegBit != 3) {
      if(fabs(muon.pullX(station,detector,arbitrationType,1)) > maxAbsPullX &&
	 fabs(muon.dX(station,detector,arbitrationType)) > maxAbsDx)
	goodMuon = false;
      if(fabs(muon.pullY(station,detector,arbitrationType,1)) > maxAbsPullY &&
	 fabs(muon.dY(station,detector,arbitrationType)) > maxAbsDy)
	goodMuon = false;
   } else {
      // special consideration for dt where there is no y information in station 4
      // impose y cuts on next station with segment
      if(fabs(muon.pullX(4,1,arbitrationType,1)) > maxAbsPullX &&
	 fabs(muon.dX(4,1,arbitrationType)) > maxAbsDx)
	goodMuon = false;
      if(theStationMask & 1<<2) {
	 if(fabs(muon.pullY(3,1,arbitrationType,1)) > maxAbsPullY &&
	    fabs(muon.dY(3,1,arbitrationType)) > maxAbsDy)
	   goodMuon = false;
      } else if(theStationMask & 1<<1) {
	 if(fabs(muon.pullY(2,1,arbitrationType,1)) > maxAbsPullY &&
	    fabs(muon.dY(2,1,arbitrationType)) > maxAbsDy)
	   goodMuon = false;
      } else if(theStationMask & 1<<0) {
	 if(fabs(muon.pullY(1,1,arbitrationType,1)) > maxAbsPullY &&
	    fabs(muon.dY(1,1,arbitrationType)) > maxAbsDy)
	   goodMuon = false;
      }
   }
   
   return goodMuon;
}

bool muonid::isGoodMuon( const reco::Muon& muon, SelectionType type )
{
  switch (type)
     {
      case TMLastStationLoose:
	return isGoodMuon(muon,TMLastStation,2,3,3,9999,9999,-3,-3,reco::Muon::SegmentAndTrackArbitration);
	break;
      case TMLastStationTight:
	return isGoodMuon(muon,TMLastStation,2,3,3,3,3,-3,-3,reco::Muon::SegmentAndTrackArbitration);
	break;
      default:
	return false;
     }
}
