#include "RecoMET/METAlgorithms/interface/CSCHaloAlgo.h"
/*
  [class]:  CSCHaloAlgo
  [authors]: R. Remington, The University of Florida
  [description]: See CSCHaloAlgo.h
  [date]: October 15, 2009
*/
using namespace reco;
using namespace std;
using namespace edm;
#include "TMath.h"

reco::CSCHaloData CSCHaloAlgo::Calculate(const CSCGeometry& TheCSCGeometry ,edm::Handle<reco::TrackCollection>& TheCSCTracks, edm::Handle<CSCSegmentCollection>& TheCSCSegments, edm::Handle<CSCRecHit2DCollection>& TheCSCRecHits,edm::Handle < L1MuGMTReadoutCollection >& TheL1GMTReadout)
{
  reco::CSCHaloData TheCSCHaloData;
  if( TheCSCTracks.isValid() )
    {
      for( reco::TrackCollection::const_iterator iTrack = TheCSCTracks->begin() ; iTrack != TheCSCTracks->end() ; iTrack++ )
	{
	  bool StoreTrack = false;
	  // Calculate global phi coordinate for central most rechit in the track
	  //float global_phi = 0.;
	  float global_z = 1200.;
	  GlobalPoint ClosestGlobalPosition;
	  for(unsigned int j = 0 ; j < iTrack->extra()->recHits().size(); j++ )
	    {
	      edm::Ref<TrackingRecHitCollection> hit( iTrack->extra()->recHits(), j );
	      DetId TheDetUnitId(hit->geographicalId());
	      if( TheDetUnitId.det() != DetId::Muon ) continue;
	      if( TheDetUnitId.subdetId() != MuonSubdetId::CSC ) continue;

	      //Its a CSC Track, store it
	      StoreTrack = true;

	      const GeomDetUnit *TheUnit = TheCSCGeometry.idToDetUnit(TheDetUnitId);
	      LocalPoint TheLocalPosition = hit->localPosition();  
	      const BoundPlane& TheSurface = TheUnit->surface();
	      const GlobalPoint TheGlobalPosition = TheSurface.toGlobal(TheLocalPosition);

	      float z = TheGlobalPosition.z();
	      if( TMath::Abs(z) < global_z )
		{
		  global_z = TMath::Abs(z);
		  ClosestGlobalPosition = GlobalPoint( TheGlobalPosition);
		}
	    }
	  TheCSCHaloData.GetCSCTrackImpactPositions().push_back(ClosestGlobalPosition);
	  
	  if( StoreTrack )
	    {
	      edm::Ref<TrackCollection> TheTrackRef( TheCSCTracks, iTrack - TheCSCTracks->begin() ) ;
	      TheCSCHaloData.GetTracks().push_back( TheTrackRef );
	    }
	}
    }
   if( TheCSCSegments.isValid() )
    {
      for(CSCSegmentCollection::const_iterator iSegment = TheCSCSegments->begin(); iSegment != TheCSCSegments->end(); iSegment++) 
	{
	}    
    }
   if( TheCSCRecHits.isValid() )
     {
       for(CSCRecHit2DCollection::const_iterator iCSCRecHit = TheCSCRecHits->begin();   iCSCRecHit != TheCSCRecHits->end(); iCSCRecHit++ )
	 {
	 }
     }

   if( TheL1GMTReadout.isValid() )
     {
       L1MuGMTReadoutCollection const *gmtrc = TheL1GMTReadout.product ();
       std::vector < L1MuGMTReadoutRecord > gmt_records = gmtrc->getRecords ();
       std::vector < L1MuGMTReadoutRecord >::const_iterator igmtrr;
       
       int icsc = 0;
       int PlusZ = 0 ;
       int MinusZ = 0 ;
       // Check to see if CSC BeamHalo trigger is tripped
       for (igmtrr = gmt_records.begin (); igmtrr != gmt_records.end (); igmtrr++)
	 {
	   std::vector < L1MuRegionalCand >::const_iterator iter1;
	   std::vector < L1MuRegionalCand > rmc;
	   rmc = igmtrr->getCSCCands ();
	   for (iter1 = rmc.begin (); iter1 != rmc.end (); iter1++)
	     {
	      if (!(*iter1).empty ())
		{
		  if ((*iter1).isFineHalo ())
		    {
		      if( (*iter1).etaValue() > 0 )
			PlusZ++;
		      else
			MinusZ++;
		    }
		  else
		    icsc++;
		}
	     }
	 }
       TheCSCHaloData.SetNumberOfHaloTriggers(PlusZ, MinusZ);
     }
   return TheCSCHaloData;
}


