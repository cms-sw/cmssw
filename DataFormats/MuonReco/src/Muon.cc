#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

using namespace reco;

Muon::Muon(  Charge q, const LorentzVector & p4, const Point & vtx ) :
  RecoCandidate( q, p4, vtx, -13 * q ) {
     energyValid_  = false;
     matchesValid_ = false;
     isolationValid_ = false;
     pfIsolationValid_ = false;
     qualityValid_ = false;
     caloCompatibility_ = -9999.;
     type_ = 0;
     bestTrackType_ = reco::Muon::None;
     bestPFTrackType_ = reco::Muon::None;
}

Muon::Muon() {
   energyValid_  = false;
   matchesValid_ = false;
   isolationValid_ = false;
   pfIsolationValid_ = false;
   qualityValid_ = false;
   caloCompatibility_ = -9999.;
   type_ = 0;
   bestTrackType_ = reco::Muon::None;
   bestPFTrackType_ = reco::Muon::None;

}

bool Muon::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && 
	   ( checkOverlap( track(), o->track() ) ||
	     checkOverlap( standAloneMuon(), o->standAloneMuon() ) ||
	     checkOverlap( combinedMuon(), o->combinedMuon() ) ||
	     checkOverlap( standAloneMuon(), o->track() ) ||
	     checkOverlap( combinedMuon(), o->track() ) )
	   );
}

Muon * Muon::clone() const {
  return new Muon( * this );
}

int Muon::numberOfChambersNoRPC() const
{
  int total = 0;
  int nAll = numberOfChambers();
  for (int iC = 0; iC < nAll; ++iC){
    if (matches()[iC].detector() == MuonSubdetId::RPC) continue;
    total++;
  }

  return total;
}

int Muon::numberOfMatches( ArbitrationType type ) const
{
   int matches(0);
   for( std::vector<MuonChamberMatch>::const_iterator chamberMatch = muMatches_.begin();
         chamberMatch != muMatches_.end(); chamberMatch++ )
   {
      if(type == RPCHitAndTrackArbitration) {
         if(chamberMatch->rpcMatches.empty()) continue;
         matches += chamberMatch->rpcMatches.size();
         continue;
      }

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
	 if(type == SegmentAndTrackArbitrationCleaned)
	   if(segmentMatch->isMask(MuonSegmentMatch::BestInChamberByDR) &&
	         segmentMatch->isMask(MuonSegmentMatch::BelongsToTrackByDR) && 
	         segmentMatch->isMask(MuonSegmentMatch::BelongsToTrackByCleaning)) {
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

int Muon::numberOfMatchedStations( ArbitrationType type ) const
{
   int stations(0);

   unsigned int theStationMask = stationMask(type);
   // eight stations, eight bits
   for(int it = 0; it < 8; ++it)
      if (theStationMask & 1<<it)
         ++stations;

   return stations;
}

unsigned int Muon::stationMask( ArbitrationType type ) const
{
   unsigned int totMask(0);
   unsigned int curMask(0);

   for( std::vector<MuonChamberMatch>::const_iterator chamberMatch = muMatches_.begin();
         chamberMatch != muMatches_.end(); chamberMatch++ )
   {
      if(type == RPCHitAndTrackArbitration) {
	 if(chamberMatch->rpcMatches.empty()) continue;

	 RPCDetId rollId = chamberMatch->id.rawId();
	 const int region    = rollId.region();
	 int rpcIndex = 1; if (region!=0) rpcIndex = 2;

         for( std::vector<MuonRPCHitMatch>::const_iterator rpcMatch = chamberMatch->rpcMatches.begin();
               rpcMatch != chamberMatch->rpcMatches.end(); rpcMatch++ )
         {
            curMask = 1<<( (chamberMatch->station()-1)+4*(rpcIndex-1) );

            // do not double count
            if(!(totMask & curMask))
               totMask += curMask;
         }
         continue;
      }

      if(chamberMatch->segmentMatches.empty()) continue;
      if(type == NoArbitration) {
         curMask = 1<<( (chamberMatch->station()-1)+4*(chamberMatch->detector()-1) );
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
               curMask = 1<<( (chamberMatch->station()-1)+4*(chamberMatch->detector()-1) );
               // do not double count
               if(!(totMask & curMask))
                  totMask += curMask;
               break;
            }
         if(type == SegmentAndTrackArbitration)
            if(segmentMatch->isMask(MuonSegmentMatch::BestInStationByDR) &&
                  segmentMatch->isMask(MuonSegmentMatch::BelongsToTrackByDR)) {
               curMask = 1<<( (chamberMatch->station()-1)+4*(chamberMatch->detector()-1) );
               // do not double count
               if(!(totMask & curMask))
                  totMask += curMask;
               break;
            }
	 if(type == SegmentAndTrackArbitrationCleaned)
            if(segmentMatch->isMask(MuonSegmentMatch::BestInStationByDR) &&
	          segmentMatch->isMask(MuonSegmentMatch::BelongsToTrackByDR) &&
	          segmentMatch->isMask(MuonSegmentMatch::BelongsToTrackByCleaning)) {
               curMask = 1<<( (chamberMatch->station()-1)+4*(chamberMatch->detector()-1) );
               // do not double count
               if(!(totMask & curMask))
                  totMask += curMask;
               break;
            }
         if(type > 1<<7)
            if(segmentMatch->isMask(type)) {
               curMask = 1<<( (chamberMatch->station()-1)+4*(chamberMatch->detector()-1) );
               // do not double count
               if(!(totMask & curMask))
                  totMask += curMask;
               break;
            }
      }
   }

   return totMask;
}

int Muon::numberOfMatchedRPCLayers( ArbitrationType type ) const
{
   int layers(0);

   unsigned int theRPCLayerMask = RPClayerMask(type);
   // maximum ten layers because of 6 layers in barrel and 3 (4) layers in each endcap before (after) upscope
   for(int it = 0; it < 10; ++it)
     if (theRPCLayerMask & 1<<it)
       ++layers;

   return layers;
}

unsigned int Muon::RPClayerMask( ArbitrationType type ) const
{
   unsigned int totMask(0);
   unsigned int curMask(0);
   for( std::vector<MuonChamberMatch>::const_iterator chamberMatch = muMatches_.begin();
	 chamberMatch != muMatches_.end(); chamberMatch++ )
   {
      if(chamberMatch->rpcMatches.empty()) continue;
	 
      RPCDetId rollId = chamberMatch->id.rawId();
      const int region = rollId.region();

      const int layer  = rollId.layer();
      int rpcLayer = chamberMatch->station();
      if (region==0) {
	 rpcLayer = chamberMatch->station()-1 + chamberMatch->station()*layer;
	 if ((chamberMatch->station()==2 && layer==2) || (chamberMatch->station()==4 && layer==1)) rpcLayer -= 1;
      } else rpcLayer += 6;
	 
      for( std::vector<MuonRPCHitMatch>::const_iterator rpcMatch = chamberMatch->rpcMatches.begin();
	    rpcMatch != chamberMatch->rpcMatches.end(); rpcMatch++ )
      {
	 curMask = 1<<(rpcLayer-1);

	 // do not double count
	 if(!(totMask & curMask))
	    totMask += curMask;
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
               curMask = 1<<( (stationIndex-1)+4*(detectorIndex-1) );
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
               curMask = 1<<((stationIndex-1)+4*(detectorIndex-1));
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
	 if(type == SegmentAndTrackArbitrationCleaned)
            if(segmentMatch->isMask(MuonSegmentMatch::BestInStationByDR) &&
                  segmentMatch->isMask(MuonSegmentMatch::BelongsToTrackByDR) &&
	          segmentMatch->isMask(MuonSegmentMatch::BelongsToTrackByCleaning)) {
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

const std::vector<const MuonChamberMatch*> Muon::chambers( int station, int muonSubdetId ) const
{
   std::vector<const MuonChamberMatch*> chambers;
   for(std::vector<MuonChamberMatch>::const_iterator chamberMatch = muMatches_.begin();
         chamberMatch != muMatches_.end(); chamberMatch++)
      if(chamberMatch->station()==station && chamberMatch->detector()==muonSubdetId)
         chambers.push_back(&(*chamberMatch));
   return chambers;
}

std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> Muon::pair( const std::vector<const MuonChamberMatch*> &chambers,
     ArbitrationType type ) const
{
   MuonChamberMatch* m = 0;
   MuonSegmentMatch* s = 0;
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair(m,s);

   if(chambers.empty()) return chamberSegmentPair;
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
	 if(type == SegmentAndTrackArbitrationCleaned)
            if(segmentMatch->isMask(MuonSegmentMatch::BestInStationByDR) &&
	          segmentMatch->isMask(MuonSegmentMatch::BelongsToTrackByDR) &&
	          segmentMatch->isMask(MuonSegmentMatch::BelongsToTrackByCleaning))
               return std::make_pair(*chamberMatch, &(*segmentMatch));
         if(type > 1<<7)
            if(segmentMatch->isMask(type))
               return std::make_pair(*chamberMatch, &(*segmentMatch));
      }
   }

   return chamberSegmentPair;
}

float Muon::dX( int station, int muonSubdetId, ArbitrationType type ) const
{
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(chambers(station,muonSubdetId),type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) return 999999;
   if(! chamberSegmentPair.second->hasPhi()) return 999999;
   return chamberSegmentPair.first->x-chamberSegmentPair.second->x;
}

float Muon::dY( int station, int muonSubdetId, ArbitrationType type ) const
{
   if(station==4 && muonSubdetId==MuonSubdetId::DT) return 999999; // no y information
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(chambers(station,muonSubdetId),type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) return 999999;
   if(! chamberSegmentPair.second->hasZed()) return 999999;
   return chamberSegmentPair.first->y-chamberSegmentPair.second->y;
}

float Muon::dDxDz( int station, int muonSubdetId, ArbitrationType type ) const
{
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(chambers(station,muonSubdetId),type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) return 999999;
   if(! chamberSegmentPair.second->hasPhi()) return 999999;
   return chamberSegmentPair.first->dXdZ-chamberSegmentPair.second->dXdZ;
}

float Muon::dDyDz( int station, int muonSubdetId, ArbitrationType type ) const
{
   if(station==4 && muonSubdetId==MuonSubdetId::DT) return 999999; // no y information
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(chambers(station,muonSubdetId),type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) return 999999;
   if(! chamberSegmentPair.second->hasZed()) return 999999;
   return chamberSegmentPair.first->dYdZ-chamberSegmentPair.second->dYdZ;
}

float Muon::pullX( int station, int muonSubdetId, ArbitrationType type, bool includeSegmentError ) const
{
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(chambers(station,muonSubdetId),type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) return 999999;
   if(! chamberSegmentPair.second->hasPhi()) return 999999;
   if(includeSegmentError)
      return (chamberSegmentPair.first->x-chamberSegmentPair.second->x)/sqrt(pow(chamberSegmentPair.first->xErr,2)+pow(chamberSegmentPair.second->xErr,2));
   return (chamberSegmentPair.first->x-chamberSegmentPair.second->x)/chamberSegmentPair.first->xErr;
}

float Muon::pullY( int station, int muonSubdetId, ArbitrationType type, bool includeSegmentError) const
{
   if(station==4 && muonSubdetId==MuonSubdetId::DT) return 999999; // no y information
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(chambers(station,muonSubdetId),type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) return 999999;
   if(! chamberSegmentPair.second->hasZed()) return 999999;
   if(includeSegmentError)
      return (chamberSegmentPair.first->y-chamberSegmentPair.second->y)/sqrt(pow(chamberSegmentPair.first->yErr,2)+pow(chamberSegmentPair.second->yErr,2));
   return (chamberSegmentPair.first->y-chamberSegmentPair.second->y)/chamberSegmentPair.first->yErr;
}

float Muon::pullDxDz( int station, int muonSubdetId, ArbitrationType type, bool includeSegmentError ) const
{
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(chambers(station,muonSubdetId),type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) return 999999;
   if(! chamberSegmentPair.second->hasPhi()) return 999999;
   if(includeSegmentError)
      return (chamberSegmentPair.first->dXdZ-chamberSegmentPair.second->dXdZ)/sqrt(pow(chamberSegmentPair.first->dXdZErr,2)+pow(chamberSegmentPair.second->dXdZErr,2));
   return (chamberSegmentPair.first->dXdZ-chamberSegmentPair.second->dXdZ)/chamberSegmentPair.first->dXdZErr;
}

float Muon::pullDyDz( int station, int muonSubdetId, ArbitrationType type, bool includeSegmentError ) const
{
   if(station==4 && muonSubdetId==MuonSubdetId::DT) return 999999; // no y information
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(chambers(station,muonSubdetId),type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) return 999999;
   if(! chamberSegmentPair.second->hasZed()) return 999999;
   if(includeSegmentError)
      return (chamberSegmentPair.first->dYdZ-chamberSegmentPair.second->dYdZ)/sqrt(pow(chamberSegmentPair.first->dYdZErr,2)+pow(chamberSegmentPair.second->dYdZErr,2));
   return (chamberSegmentPair.first->dYdZ-chamberSegmentPair.second->dYdZ)/chamberSegmentPair.first->dYdZErr;
}

float Muon::segmentX( int station, int muonSubdetId, ArbitrationType type ) const
{
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(chambers(station,muonSubdetId),type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) return 999999;
   if(! chamberSegmentPair.second->hasPhi()) return 999999;
   return chamberSegmentPair.second->x;
}

float Muon::segmentY( int station, int muonSubdetId, ArbitrationType type ) const
{
   if(station==4 && muonSubdetId==MuonSubdetId::DT) return 999999; // no y information
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(chambers(station,muonSubdetId),type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) return 999999;
   if(! chamberSegmentPair.second->hasZed()) return 999999;
   return chamberSegmentPair.second->y;
}

float Muon::segmentDxDz( int station, int muonSubdetId, ArbitrationType type ) const
{
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(chambers(station,muonSubdetId),type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) return 999999;
   if(! chamberSegmentPair.second->hasPhi()) return 999999;
   return chamberSegmentPair.second->dXdZ;
}

float Muon::segmentDyDz( int station, int muonSubdetId, ArbitrationType type ) const
{
   if(station==4 && muonSubdetId==MuonSubdetId::DT) return 999999; // no y information
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(chambers(station,muonSubdetId),type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) return 999999;
   if(! chamberSegmentPair.second->hasZed()) return 999999;
   return chamberSegmentPair.second->dYdZ;
}

float Muon::segmentXErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(chambers(station,muonSubdetId),type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) return 999999;
   if(! chamberSegmentPair.second->hasPhi()) return 999999;
   return chamberSegmentPair.second->xErr;
}

float Muon::segmentYErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   if(station==4 && muonSubdetId==MuonSubdetId::DT) return 999999; // no y information
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(chambers(station,muonSubdetId),type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) return 999999;
   if(! chamberSegmentPair.second->hasZed()) return 999999;
   return chamberSegmentPair.second->yErr;
}

float Muon::segmentDxDzErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(chambers(station,muonSubdetId),type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) return 999999;
   if(! chamberSegmentPair.second->hasPhi()) return 999999;
   return chamberSegmentPair.second->dXdZErr;
}

float Muon::segmentDyDzErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   if(station==4 && muonSubdetId==MuonSubdetId::DT) return 999999; // no y information
   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(chambers(station,muonSubdetId),type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) return 999999;
   if(! chamberSegmentPair.second->hasZed()) return 999999;
   return chamberSegmentPair.second->dYdZErr;
}

float Muon::trackEdgeX( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
   if(muonChambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers,type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) {
      float dist  = 999999;
      float supVar = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator muonChamber = muonChambers.begin();
            muonChamber != muonChambers.end(); ++muonChamber) {
         float currDist = (*muonChamber)->dist();
         if(currDist<dist) {
            dist  = currDist;
            supVar = (*muonChamber)->edgeX;
         }
      }
      return supVar;
   } else return chamberSegmentPair.first->edgeX;
}

float Muon::trackEdgeY( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
   if(muonChambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers,type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) {
      float dist  = 999999;
      float supVar = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator muonChamber = muonChambers.begin();
            muonChamber != muonChambers.end(); ++muonChamber) {
         float currDist = (*muonChamber)->dist();
         if(currDist<dist) {
            dist  = currDist;
            supVar = (*muonChamber)->edgeY;
         }
      }
      return supVar;
   } else return chamberSegmentPair.first->edgeY;
}

float Muon::trackX( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
   if(muonChambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers,type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) {
      float dist  = 999999;
      float supVar = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator muonChamber = muonChambers.begin();
            muonChamber != muonChambers.end(); ++muonChamber) {
         float currDist = (*muonChamber)->dist();
         if(currDist<dist) {
            dist  = currDist;
            supVar = (*muonChamber)->x;
         }
      }
      return supVar;
   } else return chamberSegmentPair.first->x;
}

float Muon::trackY( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
   if(muonChambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers,type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) {
      float dist  = 999999;
      float supVar = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator muonChamber = muonChambers.begin();
            muonChamber != muonChambers.end(); ++muonChamber) {
         float currDist = (*muonChamber)->dist();
         if(currDist<dist) {
            dist  = currDist;
            supVar = (*muonChamber)->y;
         }
      }
      return supVar;
   } else return chamberSegmentPair.first->y;
}

float Muon::trackDxDz( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
   if(muonChambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers,type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) {
      float dist  = 999999;
      float supVar = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator muonChamber = muonChambers.begin();
            muonChamber != muonChambers.end(); ++muonChamber) {
         float currDist = (*muonChamber)->dist();
         if(currDist<dist) {
            dist  = currDist;
            supVar = (*muonChamber)->dXdZ;
         }
      }
      return supVar;
   } else return chamberSegmentPair.first->dXdZ;
}

float Muon::trackDyDz( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
   if(muonChambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers,type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) {
      float dist  = 999999;
      float supVar = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator muonChamber = muonChambers.begin();
            muonChamber != muonChambers.end(); ++muonChamber) {
         float currDist = (*muonChamber)->dist();
         if(currDist<dist) {
            dist  = currDist;
            supVar = (*muonChamber)->dYdZ;
         }
      }
      return supVar;
   } else return chamberSegmentPair.first->dYdZ;
}

float Muon::trackXErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
   if(muonChambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers,type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) {
      float dist  = 999999;
      float supVar = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator muonChamber = muonChambers.begin();
            muonChamber != muonChambers.end(); ++muonChamber) {
         float currDist = (*muonChamber)->dist();
         if(currDist<dist) {
            dist  = currDist;
            supVar = (*muonChamber)->xErr;
         }
      }
      return supVar;
   } else return chamberSegmentPair.first->xErr;
}

float Muon::trackYErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
   if(muonChambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers,type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) {
      float dist  = 999999;
      float supVar = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator muonChamber = muonChambers.begin();
            muonChamber != muonChambers.end(); ++muonChamber) {
         float currDist = (*muonChamber)->dist();
         if(currDist<dist) {
            dist  = currDist;
            supVar = (*muonChamber)->yErr;
         }
      }
      return supVar;
   } else return chamberSegmentPair.first->yErr;
}

float Muon::trackDxDzErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
   if(muonChambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers,type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) {
      float dist  = 999999;
      float supVar = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator muonChamber = muonChambers.begin();
            muonChamber != muonChambers.end(); ++muonChamber) {
         float currDist = (*muonChamber)->dist();
         if(currDist<dist) {
            dist  = currDist;
            supVar = (*muonChamber)->dXdZErr;
         }
      }
      return supVar;
   } else return chamberSegmentPair.first->dXdZErr;
}

float Muon::trackDyDzErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
   if(muonChambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers,type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) {
      float dist  = 999999;
      float supVar = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator muonChamber = muonChambers.begin();
            muonChamber != muonChambers.end(); ++muonChamber) {
         float currDist = (*muonChamber)->dist();
         if(currDist<dist) {
            dist  = currDist;
            supVar = (*muonChamber)->dYdZErr;
         }
      }
      return supVar;
   } else return chamberSegmentPair.first->dYdZErr;
}

float Muon::trackDist( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
   if(muonChambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers,type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) {
      float dist  = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator muonChamber = muonChambers.begin();
            muonChamber != muonChambers.end(); ++muonChamber) {
         float currDist = (*muonChamber)->dist();
         if(currDist<dist) dist  = currDist;
      }
      return dist;
   } else return chamberSegmentPair.first->dist();
}

float Muon::trackDistErr( int station, int muonSubdetId, ArbitrationType type ) const
{
   const std::vector<const MuonChamberMatch*> muonChambers = chambers(station, muonSubdetId);
   if(muonChambers.empty()) return 999999;

   std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> chamberSegmentPair = pair(muonChambers,type);
   if(chamberSegmentPair.first==0 || chamberSegmentPair.second==0) {
      float dist  = 999999;
      float supVar = 999999;
      for(std::vector<const MuonChamberMatch*>::const_iterator muonChamber = muonChambers.begin();
            muonChamber != muonChambers.end(); ++muonChamber) {
         float currDist = (*muonChamber)->dist();
         if(currDist<dist) {
            dist  = currDist;
            supVar = (*muonChamber)->distErr();
         }
      }
      return supVar;
   } else return chamberSegmentPair.first->distErr();
}

void Muon::setIsolation( const MuonIsolation& isoR03, const MuonIsolation& isoR05 )
{ 
   isolationR03_ = isoR03;
   isolationR05_ = isoR05;
   isolationValid_ = true; 
}


void Muon::setPFIsolation(const std::string& label, const MuonPFIsolation& deposit) 
{ 
  if(label=="pfIsolationR03")
    pfIsolationR03_ = deposit;

  if(label=="pfIsolationR04")
    pfIsolationR04_ = deposit;

  if(label=="pfIsoMeanDRProfileR03")
    pfIsoMeanDRR03_ = deposit;

  if(label=="pfIsoMeanDRProfileR04")
    pfIsoMeanDRR04_ = deposit;

  if(label=="pfIsoSumDRProfileR03")
    pfIsoSumDRR03_ = deposit;

  if(label=="pfIsoSumDRProfileR04")
    pfIsoSumDRR04_ = deposit;

   pfIsolationValid_ = true; 
}


void Muon::setPFP4( const reco::Candidate::LorentzVector& p4 )
{ 
    pfP4_ = p4;
    type_ = type_ | PFMuon;
}



void Muon::setOuterTrack( const TrackRef & t ) { outerTrack_ = t; }
void Muon::setInnerTrack( const TrackRef & t ) { innerTrack_ = t; }
void Muon::setTrack( const TrackRef & t ) { setInnerTrack(t); }
void Muon::setStandAlone( const TrackRef & t ) { setOuterTrack(t); }
void Muon::setGlobalTrack( const TrackRef & t ) { globalTrack_ = t; }
void Muon::setCombined( const TrackRef & t ) { setGlobalTrack(t); }


bool Muon::isAValidMuonTrack(const MuonTrackType& type) const{
  return muonTrack(type).isNonnull();
}

TrackRef Muon::muonTrack(const MuonTrackType& type) const{
  switch (type) {
  case InnerTrack:     return innerTrack();
  case OuterTrack:     return standAloneMuon(); 
  case CombinedTrack:  return globalTrack();
  case TPFMS:          return tpfmsTrack();
  case Picky:          return pickyTrack();
  case DYT:            return dytTrack();
  default:             return muonTrackFromMap(type);
  }
}

void Muon::setMuonTrack(const MuonTrackType& type, const TrackRef& t) {
  
  switch (type) {
  case InnerTrack:    setInnerTrack(t);             break;
  case OuterTrack:    setStandAlone(t);             break;
  case CombinedTrack: setGlobalTrack(t);            break;
  default:            refittedTrackMap_[type] = t;  break;
  }

}

