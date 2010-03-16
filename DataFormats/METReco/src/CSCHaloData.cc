#include "DataFormats/METReco/interface/CSCHaloData.h"

/*
  [class]:  CSCHaloData
  [authors]: R. Remington, The University of Florida
  [description]: See CSCHaloData.h 
  [date]: October 15, 2009
*/

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cone.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutRecord.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"


using namespace reco;
CSCHaloData::CSCHaloData()
{
  nTriggers_PlusZ = 0;
  nTriggers_MinusZ = 0 ;
  nTracks_PlusZ = 0 ;
  nTracks_MinusZ = 0;
  HLTAccept=false;

  nOutOfTimeTriggers_PlusZ=0;
  nOutOfTimeTriggers_MinusZ=0;
  nOutOfTimeHits = 0 ;
}

int CSCHaloData::NumberOfHaloTriggers(int z) const
{
  if( z == 1 )
    return nTriggers_PlusZ;
  else if( z == -1 )
    return nTriggers_MinusZ;
  else 
    return nTriggers_MinusZ + nTriggers_PlusZ;
}

short int CSCHaloData::NumberOfOutOfTimeTriggers(int z ) const
{
  if( z == 1 ) 
    return nOutOfTimeTriggers_PlusZ;
  else if( z == -1 ) 
    return nOutOfTimeTriggers_MinusZ;
  else
    return nOutOfTimeTriggers_PlusZ+nOutOfTimeTriggers_MinusZ;
}

int CSCHaloData::NumberOfHaloTracks(int z) const 
{
  int n = 0 ;
  for(unsigned int i = 0 ; i < TheTrackRefs.size() ; i++ )
    {
      edm::Ref<reco::TrackCollection> iTrack( TheTrackRefs, i ) ;
      // Does the track go through both endcaps ? 
      bool Traversing =  (iTrack->outerPosition().z() > 0 &&  iTrack->innerPosition().z() < 0) ||  (iTrack->outerPosition().z() < 0 &&  iTrack->innerPosition().z() > 0);
      // Does the track go through only +Z endcap ?
      bool PlusZ =  (iTrack->outerPosition().z() > 0 && iTrack->innerPosition().z() > 0 ) ;
      // Does the track go through only -Z endcap ? 
      bool MinusZ = (iTrack->outerPosition().z()< 0 && iTrack->innerPosition().z() < 0) ;

      if( (z == 1) && ( PlusZ || Traversing) ) 
	n++;
      else if( (z == -1) && ( MinusZ || Traversing ) )
	n++;
      else if( (TMath::Abs(z) != 1) && (PlusZ || MinusZ || Traversing) ) 
	n++ ;
    }
  return n;
}
