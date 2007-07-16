#include "RecoMuon/MuonIdentification/interface/TrackerMuonIdentification.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void TrackerMuonIdentification::setParameters(const edm::ParameterSet& iConfig)
{
   type_ = LastStation; // default type
   std::string type = iConfig.getParameter<std::string>("algorithmType");
   if ( type.compare("LastStation") != 0 )
     edm::LogWarning("MuonIdentification") << "Unknown algorithm type is requested: " << type << "\nUsing the default one.";
   
   minPt_                   = iConfig.getParameter<double>("minPt");
   minP_                    = iConfig.getParameter<double>("minP");
   minNumberOfMatches_      = iConfig.getParameter<int>("minNumberOfMatchedStations");
   maxAbsDx_                = iConfig.getParameter<double>("maxAbsDx");
   maxAbsPullX_             = iConfig.getParameter<double>("maxAbsPullX");
   maxAbsDy_                = iConfig.getParameter<double>("maxAbsDy");
   maxAbsPullY_             = iConfig.getParameter<double>("maxAbsPullY");
   maxChamberDist_          = iConfig.getParameter<double>("maxChamberDistance");
   
   std::string arbitrationType = iConfig.getParameter<std::string>("arbitrationType");
   if (arbitrationType.compare("NoArbitration")==0)
     arbitrationType_ = reco::Muon::NoArbitration;
   else if (arbitrationType.compare("SegmentArbitration")==0)
     arbitrationType_ = reco::Muon::SegmentArbitration;
   else if (arbitrationType.compare("SegmentAndTrackArbitration")==0)
     arbitrationType_ = reco::Muon::SegmentAndTrackArbitration;
   else {
      edm::LogWarning("MuonIdentification") << "Unknown arbitration type is requested: " << arbitrationType << "\nUsing the default one";
      arbitrationType_ = reco::Muon::SegmentAndTrackArbitration;
   }
}

unsigned int TrackerMuonIdentification::RequiredStationMask( const reco::Muon& muon ) const
{
   int theMask = 0;

   for(int stationIdx = 1; stationIdx < 5; ++stationIdx)
      for(int detectorIdx = 1; detectorIdx < 3; ++detectorIdx)
         if(muon.trackDist(stationIdx,detectorIdx,arbitrationType_) < 9E5 &&
               muon.trackDist(stationIdx,detectorIdx,arbitrationType_)/muon.trackDistErr(stationIdx,detectorIdx,arbitrationType_) < maxChamberDist_)
            theMask += 1<<((stationIdx-1)+4*(detectorIdx-1));

   return theMask;
}

bool TrackerMuonIdentification::isGoodMuon( const reco::Muon& muon) const
{
   if (!muon.isMatchesValid()) return false;
   bool goodMuon = false;

   unsigned int theStationMask = muon.stationMask(arbitrationType_);
   unsigned int theRequiredStationMask = RequiredStationMask(muon);

   // If there are no required stations, require there be at least two segments
   int numSegs = 0;
   for(int it = 0; it < 8; ++it)
      numSegs += theStationMask & 1<<it;

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
      if(fabs(muon.pullX(station,detector,arbitrationType_,1)) > maxAbsPullX_ &&
	 fabs(muon.dX(station,detector,arbitrationType_)) > maxAbsDx_)
	goodMuon = false;
      if(fabs(muon.pullY(station,detector,arbitrationType_,1)) > maxAbsPullY_ &&
	 fabs(muon.dY(station,detector,arbitrationType_)) > maxAbsDy_)
	goodMuon = false;
   } else {
      // special consideration for dt where there is no y information in station 4
      // impose y cuts on next station with segment
      if(fabs(muon.pullX(4,1,arbitrationType_,1)) > maxAbsPullX_ &&
	 fabs(muon.dX(4,1,arbitrationType_)) > maxAbsDx_)
	goodMuon = false;
      if(theStationMask & 2) {
	 if(fabs(muon.pullY(3,1,arbitrationType_,1)) > maxAbsPullY_ &&
	    fabs(muon.dY(3,1,arbitrationType_)) > maxAbsDy_)
	   goodMuon = false;
      } else if(theStationMask & 1) {
	 if(fabs(muon.pullY(2,1,arbitrationType_,1)) > maxAbsPullY_ &&
	    fabs(muon.dY(2,1,arbitrationType_)) > maxAbsDy_)
	   goodMuon = false;
      } else if(theStationMask & 0) {
	 if(fabs(muon.pullY(1,1,arbitrationType_,1)) > maxAbsPullY_ &&
	    fabs(muon.dY(1,1,arbitrationType_)) > maxAbsDy_)
	   goodMuon = false;
      }
   }
   
   return goodMuon;
}
