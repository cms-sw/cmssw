#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
using namespace reco;

Muon::Muon(  Charge q, const LorentzVector & p4, const Point & vtx ) :
  RecoCandidate( q, p4, vtx, -13 * q ) {
     energyValid_  = false;
     matchesValid_ = false;
     isolationValid_ = false;
     caloCompatibility_ = -9999.;
     type_ = 0;
}

Muon::Muon() {
   energyValid_  = false;
   matchesValid_ = false;
   isolationValid_ = false;
   caloCompatibility_ = -9999.;
   type_ = 0;
}

bool Muon::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && 
	   ( checkOverlap( track(), o->track() ) ||
	     checkOverlap( standAloneMuon(), o->standAloneMuon() ) ||
	     checkOverlap( combinedMuon(), o->combinedMuon() ) ||
	     checkOverlap( superCluster(), o->superCluster() ) ) 
	   );
}

Muon * Muon::clone() const {
  return new Muon( * this );
}

int Muon::numberOfMatches( ArbitrationType type ) const
{
   int matches(0);
   for( std::vector<MuonChamberMatch>::const_iterator chamberMatch = muMatches_.begin();
         chamberMatch != muMatches_.end(); chamberMatch++ )
   {
      if(chamberMatch->segmentMatches.empty()) continue;
      if(type == NoArbitration) {
         matches++;
         continue;
      }

      for( std::vector<MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
            segmentMatch != chamberMatch->segmentMatches.end(); segmentMatch++ )
      {
         if(type == SegmentArbitration)
            if(segmentMatch->isMask(MuonSegmentMatch::BestInChamberByDR)) {
               matches++;
               break;
            }
         if(type == SegmentAndTrackArbitration)
            if(segmentMatch->isMask(MuonSegmentMatch::BestInChamberByDR) &&
                  segmentMatch->isMask(MuonSegmentMatch::BelongsToTrackByDR)) {
               matches++;
               break;
            }
         if(type > 1<<7)
            if(segmentMatch->isMask(type)) {
               matches++;
               break;
            }
      }
   }

   return matches;
}

unsigned int Muon::stationMask( ArbitrationType type ) const
{
   unsigned int totMask(0);
   unsigned int curMask(0);
   for( std::vector<MuonChamberMatch>::const_iterator chamberMatch = muMatches_.begin();
         chamberMatch != muMatches_.end(); chamberMatch++ )
   {
      if(chamberMatch->segmentMatches.empty()) continue;
      if(type == NoArbitration) {
         curMask = 1<<(chamberMatch->station()-1)+4*(chamberMatch->detector()-1);
         // do not double count
         if(!(totMask & curMask))
            totMask += curMask;
         continue;
      }

      for( std::vector<MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
            segmentMatch != chamberMatch->segmentMatches.end(); segmentMatch++ )
      {
         if(type == SegmentArbitration)
            if(segmentMatch->isMask(MuonSegmentMatch::BestInStationByDR)) {
               curMask = 1<<(chamberMatch->station()-1)+4*(chamberMatch->detector()-1);
               // do not double count
               if(!(totMask & curMask))
                  totMask += curMask;
               break;
            }
         if(type == SegmentAndTrackArbitration)
            if(segmentMatch->isMask(MuonSegmentMatch::BestInStationByDR) &&
                  segmentMatch->isMask(MuonSegmentMatch::BelongsToTrackByDR)) {
               curMask = 1<<(chamberMatch->station()-1)+4*(chamberMatch->detector()-1);
               // do not double count
               if(!(totMask & curMask))
                  totMask += curMask;
               break;
            }
         if(type > 1<<7)
            if(segmentMatch->isMask(type)) {
               curMask = 1<<(chamberMatch->station()-1)+4*(chamberMatch->detector()-1);
               // do not double count
               if(!(totMask & curMask))
                  totMask += curMask;
               break;
            }
      }
   }

   return totMask;
}

unsigned int Muon::stationGapMaskDistance( float distanceCut ) const
{
   unsigned int totMask(0);
   for( int stationIndex = 1; stationIndex < 5; stationIndex++ )
   {
      for( int detectorIndex = 1; detectorIndex < 4; detectorIndex++ )
      {
         unsigned int curMask(0);
         for( std::vector<MuonChamberMatch>::const_iterator chamberMatch = muMatches_.begin();
               chamberMatch != muMatches_.end(); chamberMatch++ )
         {
            if(!(chamberMatch->station()==stationIndex && chamberMatch->detector()==detectorIndex)) continue;

            float edgeX = chamberMatch->edgeX;
            float edgeY = chamberMatch->edgeY;
            if(edgeX<0 && fabs(edgeX)>fabs(distanceCut) &&
                  edgeY<0 && fabs(edgeY)>fabs(distanceCut)) // inside the chamber so negates all gaps for this station
            {
               curMask = 0;
               break;
            }
            if( ( fabs(edgeX) < fabs(distanceCut) && edgeY < fabs(distanceCut) ) ||
		( fabs(edgeY) < fabs(distanceCut) && edgeX < fabs(distanceCut) ) ) // inside gap
               curMask = 1<<(stationIndex-1)+4*(detectorIndex-1);
         }

         totMask += curMask; // add to total mask
      }
   }

   return totMask;
}

unsigned int Muon::stationGapMaskPull( float sigmaCut ) const
{
   unsigned int totMask(0);
   for( int stationIndex = 1; stationIndex < 5; stationIndex++ )
   {
      for( int detectorIndex = 1; detectorIndex < 4; detectorIndex++ )
      {
         unsigned int curMask(0);
         for( std::vector<MuonChamberMatch>::const_iterator chamberMatch = muMatches_.begin();
               chamberMatch != muMatches_.end(); chamberMatch++ )
         {
            if(!(chamberMatch->station()==stationIndex && chamberMatch->detector()==detectorIndex)) continue;

            float edgeX = chamberMatch->edgeX;
            float edgeY = chamberMatch->edgeY;
            float xErr  = chamberMatch->xErr+0.000001; // protect against division by zero later
            float yErr  = chamberMatch->yErr+0.000001; // protect against division by zero later
            if(edgeX<0 && fabs(edgeX/xErr)>fabs(sigmaCut) &&
                  edgeY<0 && fabs(edgeY/yErr)>fabs(sigmaCut)) // inside the chamber so negates all gaps for this station
            {
               curMask = 0;
               break;
            }
            if( ( fabs(edgeX/xErr) < fabs(sigmaCut) && edgeY/yErr < fabs(sigmaCut) ) ||
		( fabs(edgeY/yErr) < fabs(sigmaCut) && edgeX/xErr < fabs(sigmaCut) ) ) // inside gap
               curMask = 1<<(stationIndex-1)+4*(detectorIndex-1);
         }

         totMask += curMask; // add to total mask
      }
   }

   return totMask;
}

int Muon::numberOfSegments( int station, int muonSubdetId, ArbitrationType type ) const
{
   int segments(0);
   for( std::vector<MuonChamberMatch>::const_iterator chamberMatch = muMatches_.begin();
         chamberMatch != muMatches_.end(); chamberMatch++ )
   {
      if(chamberMatch->segmentMatches.empty()) continue;
      if(!(chamberMatch->station()==station && chamberMatch->detector()==muonSubdetId)) continue;

      if(type == NoArbitration) {
         segments += chamberMatch->segmentMatches.size();
         continue;
      }

      for( std::vector<MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
            segmentMatch != chamberMatch->segmentMatches.end(); segmentMatch++ )
      {
         if(type == SegmentArbitration)
            if(segmentMatch->isMask(MuonSegmentMatch::BestInStationByDR)) {
               segments++;
               break;
            }
         if(type == SegmentAndTrackArbitration)
            if(segmentMatch->isMask(MuonSegmentMatch::BestInStationByDR) &&
                  segmentMatch->isMask(MuonSegmentMatch::BelongsToTrackByDR)) {
               segments++;
               break;
            }
         if(type > 1<<7)
            if(segmentMatch->isMask(type)) {
               segments++;
               break;
            }
      }
   }

   return segments;
}

const std::vector<const MuonChamberMatch*> Muon::getChambers( int station, int muonSubdetId ) const
{
   std::vector<const MuonChamberMatch*> chambers;
   for(std::vector<MuonChamberMatch>::const_iterator chamberMatch = muMatches_.begin();
         chamberMatch != muMatches_.end(); chamberMatch++)
      if(chamberMatch->station()==station && chamberMatch->detector()==muonSubdetId)
         chambers.push_back(&(*chamberMatch));
   return chambers;
}

std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> Muon::getPair( const std::vector<const MuonChamberMatch*> chambers,
     ArbitrationType type ) const
{
   MuonChamberMatch* m = 0;
   MuonSegmentMatch* s = 0;
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair(m,s);

   if(chambers.empty()) return thePair;
   for( std::vector<const MuonChamberMatch*>::const_iterator chamberMatch = chambers.begin();
         chamberMatch != chambers.end(); chamberMatch++ )
   {
      if((*chamberMatch)->segmentMatches.empty()) continue;
      if(type == NoArbitration)
         return std::make_pair(*chamberMatch, &((*chamberMatch)->segmentMatches.front()));

      for( std::vector<MuonSegmentMatch>::const_iterator segmentMatch = (*chamberMatch)->segmentMatches.begin();
            segmentMatch != (*chamberMatch)->segmentMatches.end(); segmentMatch++ )
      {
         if(type == SegmentArbitration)
            if(segmentMatch->isMask(MuonSegmentMatch::BestInStationByDR)) 
               return std::make_pair(*chamberMatch, &(*segmentMatch));
         if(type == SegmentAndTrackArbitration)
            if(segmentMatch->isMask(MuonSegmentMatch::BestInStationByDR) &&
                  segmentMatch->isMask(MuonSegmentMatch::BelongsToTrackByDR))
               return std::make_pair(*chamberMatch, &(*segmentMatch));
         if(type > 1<<7)
            if(segmentMatch->isMask(type))
               return std::make_pair(*chamberMatch, &(*segmentMatch));
      }
   }

   return thePair;
}

float Muon::dX( int station, int muonSubdetId, ArbitrationType type ) const
{
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(getChambers(station,muonSubdetId),type);
   if(thePair.first==0 || thePair.second==0) return 999999;
   return thePair.first->x-thePair.second->x;
}

float Muon::dY( int station, int muonSubdetId, ArbitrationType type ) const
{
   if(station==4 && muonSubdetId==MuonSubdetId::DT) return 999999; // no y information
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(getChambers(station,muonSubdetId),type);
   if(thePair.first==0 || thePair.second==0) return 999999;
   return thePair.first->y-thePair.second->y;
}

float Muon::dDxDz( int station, int muonSubdetId, ArbitrationType type ) const
{
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(getChambers(station,muonSubdetId),type);
   if(thePair.first==0 || thePair.second==0) return 999999;
   return thePair.first->dXdZ-thePair.second->dXdZ;
}

float Muon::dDyDz( int station, int muonSubdetId, ArbitrationType type ) const
{
   if(station==4 && muonSubdetId==MuonSubdetId::DT) return 999999; // no y information
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(getChambers(station,muonSubdetId),type);
   if(thePair.first==0 || thePair.second==0) return 999999;
   return thePair.first->dYdZ-thePair.second->dYdZ;
}

float Muon::pullX( int station, int muonSubdetId, ArbitrationType type, bool includeSegmentError ) const
{
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(getChambers(station,muonSubdetId),type);
   if(thePair.first==0 || thePair.second==0) return 999999;
   if(includeSegmentError)
      return (thePair.first->x-thePair.second->x)/sqrt(pow(thePair.first->xErr,2)+pow(thePair.second->xErr,2));
   return (thePair.first->x-thePair.second->x)/thePair.first->xErr;
}

float Muon::pullY( int station, int muonSubdetId, ArbitrationType type, bool includeSegmentError) const
{
   if(station==4 && muonSubdetId==MuonSubdetId::DT) return 999999; // no y information
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(getChambers(station,muonSubdetId),type);
   if(thePair.first==0 || thePair.second==0) return 999999;
   if(includeSegmentError)
      return (thePair.first->y-thePair.second->y)/sqrt(pow(thePair.first->yErr,2)+pow(thePair.second->yErr,2));
   return (thePair.first->y-thePair.second->y)/thePair.first->yErr;
}

float Muon::pullDxDz( int station, int muonSubdetId, ArbitrationType type, bool includeSegmentError ) const
{
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(getChambers(station,muonSubdetId),type);
   if(thePair.first==0 || thePair.second==0) return 999999;
   if(includeSegmentError)
      return (thePair.first->dXdZ-thePair.second->dXdZ)/sqrt(pow(thePair.first->dXdZErr,2)+pow(thePair.second->dXdZErr,2));
   return (thePair.first->dXdZ-thePair.second->dXdZ)/thePair.first->dXdZErr;
}

float Muon::pullDyDz( int station, int muonSubdetId, ArbitrationType type, bool includeSegmentError ) const
{
   if(station==4 && muonSubdetId==MuonSubdetId::DT) return 999999; // no y information
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(getChambers(station,muonSubdetId),type);
   if(thePair.first==0 || thePair.second==0) return 999999;
   if(includeSegmentError)
      return (thePair.first->dYdZ-thePair.second->dYdZ)/sqrt(pow(thePair.first->dYdZErr,2)+pow(thePair.second->dYdZErr,2));
   return (thePair.first->dYdZ-thePair.second->dYdZ)/thePair.first->dYdZErr;
}

float Muon::segmentX( int station, int muonSubdetId, ArbitrationType type ) const
{
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(getChambers(station,muonSubdetId),type);
   if(thePair.first==0 || thePair.second==0) return 999999;
   return thePair.second->x;
}

float Muon::segmentY( int station, int muonSubdetId, ArbitrationType type ) const
{
   if(station==4 && muonSubdetId==MuonSubdetId::DT) return 999999; // no y information
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(getChambers(station,muonSubdetId),type);
   if(thePair.first==0 || thePair.second==0) return 999999;
   return thePair.second->y;
}

float Muon::segmentDxDz( int station, int muonSubdetId, ArbitrationType type ) const
{
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(getChambers(station,muonSubdetId),type);
   if(thePair.first==0 || thePair.second==0) return 999999;
   return thePair.second->dXdZ;
}

float Muon::segmentDyDz( int station, int muonSubdetId, ArbitrationType type ) const
{
   if(station==4 && muonSubdetId==MuonSubdetId::DT) return 999999; // no y information
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(getChambers(station,muonSubdetId),type);
   if(thePair.first==0 || thePair.second==0) return 999999;
   return thePair.second->dYdZ;
}

float Muon::segmentXErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(getChambers(station,muonSubdetId),type);
   if(thePair.first==0 || thePair.second==0) return 999999;
   return thePair.second->xErr;
}

float Muon::segmentYErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   if(station==4 && muonSubdetId==MuonSubdetId::DT) return 999999; // no y information
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(getChambers(station,muonSubdetId),type);
   if(thePair.first==0 || thePair.second==0) return 999999;
   return thePair.second->yErr;
}

float Muon::segmentDxDzErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(getChambers(station,muonSubdetId),type);
   if(thePair.first==0 || thePair.second==0) return 999999;
   return thePair.second->dXdZErr;
}

float Muon::segmentDyDzErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   if(station==4 && muonSubdetId==MuonSubdetId::DT) return 999999; // no y information
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(getChambers(station,muonSubdetId),type);
   if(thePair.first==0 || thePair.second==0) return 999999;
   return thePair.second->dYdZErr;
}

float Muon::trackEdgeX( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> chambers = getChambers(station, muonSubdetId);
   if(chambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(chambers,type);
   if(thePair.first==0 || thePair.second==0) {
      float theDist  = 999999;
      float theFloat = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator theChamber = chambers.begin();
            theChamber != chambers.end(); ++theChamber) {
         float currDist = (*theChamber)->dist();
         if(currDist<theDist) {
            theDist  = currDist;
            theFloat = (*theChamber)->edgeX;
         }
      }
      return theFloat;
   } else return thePair.first->edgeX;
}

float Muon::trackEdgeY( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> chambers = getChambers(station, muonSubdetId);
   if(chambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(chambers,type);
   if(thePair.first==0 || thePair.second==0) {
      float theDist  = 999999;
      float theFloat = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator theChamber = chambers.begin();
            theChamber != chambers.end(); ++theChamber) {
         float currDist = (*theChamber)->dist();
         if(currDist<theDist) {
            theDist  = currDist;
            theFloat = (*theChamber)->edgeY;
         }
      }
      return theFloat;
   } else return thePair.first->edgeY;
}

float Muon::trackX( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> chambers = getChambers(station, muonSubdetId);
   if(chambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(chambers,type);
   if(thePair.first==0 || thePair.second==0) {
      float theDist  = 999999;
      float theFloat = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator theChamber = chambers.begin();
            theChamber != chambers.end(); ++theChamber) {
         float currDist = (*theChamber)->dist();
         if(currDist<theDist) {
            theDist  = currDist;
            theFloat = (*theChamber)->x;
         }
      }
      return theFloat;
   } else return thePair.first->x;
}

float Muon::trackY( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> chambers = getChambers(station, muonSubdetId);
   if(chambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(chambers,type);
   if(thePair.first==0 || thePair.second==0) {
      float theDist  = 999999;
      float theFloat = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator theChamber = chambers.begin();
            theChamber != chambers.end(); ++theChamber) {
         float currDist = (*theChamber)->dist();
         if(currDist<theDist) {
            theDist  = currDist;
            theFloat = (*theChamber)->y;
         }
      }
      return theFloat;
   } else return thePair.first->y;
}

float Muon::trackDxDz( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> chambers = getChambers(station, muonSubdetId);
   if(chambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(chambers,type);
   if(thePair.first==0 || thePair.second==0) {
      float theDist  = 999999;
      float theFloat = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator theChamber = chambers.begin();
            theChamber != chambers.end(); ++theChamber) {
         float currDist = (*theChamber)->dist();
         if(currDist<theDist) {
            theDist  = currDist;
            theFloat = (*theChamber)->dXdZ;
         }
      }
      return theFloat;
   } else return thePair.first->dXdZ;
}

float Muon::trackDyDz( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> chambers = getChambers(station, muonSubdetId);
   if(chambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(chambers,type);
   if(thePair.first==0 || thePair.second==0) {
      float theDist  = 999999;
      float theFloat = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator theChamber = chambers.begin();
            theChamber != chambers.end(); ++theChamber) {
         float currDist = (*theChamber)->dist();
         if(currDist<theDist) {
            theDist  = currDist;
            theFloat = (*theChamber)->dYdZ;
         }
      }
      return theFloat;
   } else return thePair.first->dYdZ;
}

float Muon::trackXErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> chambers = getChambers(station, muonSubdetId);
   if(chambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(chambers,type);
   if(thePair.first==0 || thePair.second==0) {
      float theDist  = 999999;
      float theFloat = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator theChamber = chambers.begin();
            theChamber != chambers.end(); ++theChamber) {
         float currDist = (*theChamber)->dist();
         if(currDist<theDist) {
            theDist  = currDist;
            theFloat = (*theChamber)->xErr;
         }
      }
      return theFloat;
   } else return thePair.first->xErr;
}

float Muon::trackYErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> chambers = getChambers(station, muonSubdetId);
   if(chambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(chambers,type);
   if(thePair.first==0 || thePair.second==0) {
      float theDist  = 999999;
      float theFloat = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator theChamber = chambers.begin();
            theChamber != chambers.end(); ++theChamber) {
         float currDist = (*theChamber)->dist();
         if(currDist<theDist) {
            theDist  = currDist;
            theFloat = (*theChamber)->yErr;
         }
      }
      return theFloat;
   } else return thePair.first->yErr;
}

float Muon::trackDxDzErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> chambers = getChambers(station, muonSubdetId);
   if(chambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(chambers,type);
   if(thePair.first==0 || thePair.second==0) {
      float theDist  = 999999;
      float theFloat = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator theChamber = chambers.begin();
            theChamber != chambers.end(); ++theChamber) {
         float currDist = (*theChamber)->dist();
         if(currDist<theDist) {
            theDist  = currDist;
            theFloat = (*theChamber)->dXdZErr;
         }
      }
      return theFloat;
   } else return thePair.first->dXdZErr;
}

float Muon::trackDyDzErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> chambers = getChambers(station, muonSubdetId);
   if(chambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(chambers,type);
   if(thePair.first==0 || thePair.second==0) {
      float theDist  = 999999;
      float theFloat = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator theChamber = chambers.begin();
            theChamber != chambers.end(); ++theChamber) {
         float currDist = (*theChamber)->dist();
         if(currDist<theDist) {
            theDist  = currDist;
            theFloat = (*theChamber)->dYdZErr;
         }
      }
      return theFloat;
   } else return thePair.first->dYdZErr;
}

float Muon::trackDist( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> chambers = getChambers(station, muonSubdetId);
   if(chambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(chambers,type);
   if(thePair.first==0 || thePair.second==0) {
      float theDist  = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator theChamber = chambers.begin();
            theChamber != chambers.end(); ++theChamber) {
         float currDist = (*theChamber)->dist();
         if(currDist<theDist) theDist  = currDist;
      }
      return theDist;
   } else return thePair.first->dist();
}

float Muon::trackDistErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> chambers = getChambers(station, muonSubdetId);
   if(chambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> thePair = getPair(chambers,type);
   if(thePair.first==0 || thePair.second==0) {
      float theDist  = 999999;
      float theFloat = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator theChamber = chambers.begin();
            theChamber != chambers.end(); ++theChamber) {
         float currDist = (*theChamber)->dist();
         if(currDist<theDist) {
            theDist  = currDist;
            theFloat = (*theChamber)->distErr();
         }
      }
      return theFloat;
   } else return thePair.first->distErr();
}

void Muon::setIsolation( const MuonIsolation& isoR03, const MuonIsolation& isoR05 )
{ 
   isolationR03_ = isoR03;
   isolationR05_ = isoR05;
   isolationValid_ = true; 
}
