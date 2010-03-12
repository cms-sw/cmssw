#include "RecoMET/METAlgorithms/interface/CSCHaloAlgo.h"
#include "FWCore/Common/interface/TriggerNames.h"
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

CSCHaloAlgo::CSCHaloAlgo()
{
  deta_threshold = 0.;
  min_inner_radius = 0.;
  max_inner_radius = 9999.;
  min_outer_radius = 0.;
  max_outer_radius = 9999.;
  dphi_threshold = 999.;
  norm_chi2_threshold = 999.;
  recHit_t0=200;
  recHit_twindow=500;
  expected_BX=3;
}

reco::CSCHaloData CSCHaloAlgo::Calculate(const CSCGeometry& TheCSCGeometry ,edm::Handle<reco::TrackCollection>& TheCSCTracks, edm::Handle<CSCSegmentCollection>& TheCSCSegments, edm::Handle<CSCRecHit2DCollection>& TheCSCRecHits,edm::Handle < L1MuGMTReadoutCollection >& TheL1GMTReadout,edm::Handle<edm::TriggerResults>& TheHLTResults, const edm::TriggerNames * triggerNames, const edm::Handle<CSCALCTDigiCollection>& TheALCTs)
{
  reco::CSCHaloData TheCSCHaloData;
  if( TheCSCTracks.isValid() )
    {
      for( reco::TrackCollection::const_iterator iTrack = TheCSCTracks->begin() ; iTrack != TheCSCTracks->end() ; iTrack++ )
	{
	  bool StoreTrack = false;
	  // Calculate global phi coordinate for central most rechit in the track
	  float innermost_global_z = 1500.;
	  float outermost_global_z = 0.;
	  GlobalPoint InnerMostGlobalPosition;  // smallest abs(z)
	  GlobalPoint OuterMostGlobalPosition;  // largest abs(z)
	  
	  for(unsigned int j = 0 ; j < iTrack->extra()->recHits().size(); j++ )
	    {
	      edm::Ref<TrackingRecHitCollection> hit( iTrack->extra()->recHits(), j );
	      DetId TheDetUnitId(hit->geographicalId());
	      if( TheDetUnitId.det() != DetId::Muon ) continue;
	      if( TheDetUnitId.subdetId() != MuonSubdetId::CSC )
		 {
		   if( TheDetUnitId.subdetId() != MuonSubdetId::DT )
		     {
		       StoreTrack = false;
		       break;  // definitely, not halo
		     }
		   continue;
		 }


	      //Its a CSC Track, store it
	      StoreTrack = true;

	      const GeomDetUnit *TheUnit = TheCSCGeometry.idToDetUnit(TheDetUnitId);
	      LocalPoint TheLocalPosition = hit->localPosition();  
	      const BoundPlane& TheSurface = TheUnit->surface();
	      const GlobalPoint TheGlobalPosition = TheSurface.toGlobal(TheLocalPosition);

	      float z = TheGlobalPosition.z();
	      if( TMath::Abs(z) < innermost_global_z )
		{
		  innermost_global_z = TMath::Abs(z);
		  InnerMostGlobalPosition = GlobalPoint( TheGlobalPosition);
		}
	      if( TMath::Abs(z) > outermost_global_z )
		{
		  outermost_global_z = TMath::Abs(z);
		  OuterMostGlobalPosition = GlobalPoint( TheGlobalPosition );
		}
	    }
	  float deta = TMath::Abs( OuterMostGlobalPosition.eta() - InnerMostGlobalPosition.eta() );
	  float dphi = TMath::ACos( TMath::Cos( OuterMostGlobalPosition.phi() - InnerMostGlobalPosition.phi() ) ) ;
	  float innermost_x = InnerMostGlobalPosition.x() ;
	  float innermost_y = InnerMostGlobalPosition.y();
	  float outermost_x = OuterMostGlobalPosition.x();
	  float outermost_y = OuterMostGlobalPosition.y();
	  float innermost_r = TMath::Sqrt(innermost_x *innermost_x + innermost_y * innermost_y );
	  float outermost_r = TMath::Sqrt(outermost_x *outermost_x + outermost_y * outermost_y );
	  
	  if( deta < deta_threshold )
	    StoreTrack = false;
	  if( dphi > dphi_threshold )
	    StoreTrack = false;
	  if( innermost_r < min_inner_radius )
	    StoreTrack = false;
	  if( innermost_r > max_inner_radius )
	    StoreTrack = false;
	  if( outermost_r < min_outer_radius )
	    StoreTrack = false;
	  if( outermost_r > max_outer_radius )
	    StoreTrack  = false;
	  if( iTrack->normalizedChi2() > norm_chi2_threshold )
	    StoreTrack = false;

	  if( StoreTrack )
	    {
	      TheCSCHaloData.GetCSCTrackImpactPositions().push_back( InnerMostGlobalPosition );

	      edm::Ref<TrackCollection> TheTrackRef( TheCSCTracks, iTrack - TheCSCTracks->begin() ) ;
	      TheCSCHaloData.GetTracks().push_back( TheTrackRef );
	    }
	}
    }

   if( TheHLTResults.isValid() )
     {
       bool EventPasses = false;
       for( unsigned int index = 0 ; index < vIT_HLTBit.size(); index++)
         {
           if( vIT_HLTBit[index].label().size() )
             {
               //Get the HLT bit and check to make sure it is valid 
               unsigned int bit = triggerNames->triggerIndex( vIT_HLTBit[index].label().c_str());
               if( bit < TheHLTResults->size() )
                 {
		   //If any of the HLT names given by the user accept, then the event passes
		   if( TheHLTResults->accept( bit ) && !TheHLTResults->error( bit ) )
		     {
		       EventPasses = true;
		     }
                 }
             }
         }
       if( EventPasses )
         TheCSCHaloData.SetHLTBit(true);
       else
         TheCSCHaloData.SetHLTBit(false);
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

   // Loop over CSCALCTDigi collection to look for out-of-time chamber triggers 
   // A collision muon in real data should only have ALCTDigi::getBX() = 3 ( in MC, it will be 6 )
   // Note that there could be two ALCTs per chamber 
   short int n_alcts=0;
   if(TheALCTs.isValid())
     {
       for (CSCALCTDigiCollection::DigiRangeIterator j=TheALCTs->begin(); j!=TheALCTs->end(); j++) 
	 {
	   const CSCALCTDigiCollection::Range& range =(*j).second;
	   for (CSCALCTDigiCollection::const_iterator digiIt = range.first; digiIt!=range.second; ++digiIt)
	     {
	       if( (*digiIt).isValid() && ( (*digiIt).getBX() < expected_BX ) )
		 {
		   n_alcts++;
		 }
	     }
	 }
     }
   TheCSCHaloData.SetNOutOfTimeTriggers(n_alcts);

   // Loop over the CSCRecHit2D collection to look for out-of-time recHits
   // Out-of-time is defined as tpeak outside [t_0 + TOF - t_window, t_0 + TOF + t_window]
   // where t_0 and t_window are configurable parameters
   short int n_recHits = 0;
   if( TheCSCRecHits.isValid() )
     {
       CSCRecHit2DCollection::const_iterator dRHIter;
       for (dRHIter = TheCSCRecHits->begin(); dRHIter != TheCSCRecHits->end(); dRHIter++) 
	 {
	   if ( !((*dRHIter).isValid()) ) continue;  // only interested in valid hits
	   CSCDetId idrec = (CSCDetId)(*dRHIter).cscDetId();
	   float RHTime = (*dRHIter).tpeak();
	   LocalPoint rhitlocal = (*dRHIter).localPosition();
	   const CSCChamber* chamber = TheCSCGeometry.chamber(idrec);
	   GlobalPoint globalPosition = chamber->toGlobal(rhitlocal);
	   float globX = globalPosition.x();
	   float globY = globalPosition.y();
	   float globZ = globalPosition.z();
	   float TOF = (sqrt(globX*globX+ globY*globY + globZ*globZ))/29.9792458 ; //cm -> ns
	   if ( (RHTime < (recHit_t0 + TOF - recHit_twindow)) || (RHTime > (recHit_t0 + TOF + recHit_twindow)) )
	     {
	       n_recHits++;
	     }
	 }
     }
   TheCSCHaloData.SetNOutOfTimeHits(n_recHits);

   return TheCSCHaloData;
}


