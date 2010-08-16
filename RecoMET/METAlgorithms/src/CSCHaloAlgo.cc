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
  
  min_outer_theta = 0.;
  max_outer_theta = TMath::Pi();
  
  matching_dphi_threshold = 0.18; //radians
  matching_deta_threshold = 0.4;
  matching_dwire_threshold = 5.;
}

reco::CSCHaloData CSCHaloAlgo::Calculate(const CSCGeometry& TheCSCGeometry,
					 edm::Handle<reco::TrackCollection>& TheCSCTracks, 
					 edm::Handle<reco::MuonCollection>& TheMuons,
					 edm::Handle<CSCSegmentCollection>& TheCSCSegments, 
					 edm::Handle<CSCRecHit2DCollection>& TheCSCRecHits,
					 edm::Handle < L1MuGMTReadoutCollection >& TheL1GMTReadout,
					 edm::Handle<edm::TriggerResults>& TheHLTResults,
					 const edm::TriggerNames * triggerNames, const edm::Handle<CSCALCTDigiCollection>& TheALCTs)
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
	  float theta = iTrack->outerMomentum().theta();
	  float innermost_x = InnerMostGlobalPosition.x() ;
	  float innermost_y = InnerMostGlobalPosition.y();
	  float outermost_x = OuterMostGlobalPosition.x();
	  float outermost_y = OuterMostGlobalPosition.y();
	  float innermost_r = TMath::Sqrt(innermost_x *innermost_x + innermost_y * innermost_y );
	  float outermost_r = TMath::Sqrt(outermost_x *outermost_x + outermost_y * outermost_y );
	  
	  if( deta < deta_threshold )
	    StoreTrack = false;
	  if( theta > min_outer_theta && theta < max_outer_theta )
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
		      float halophi = iter1->phiValue();
		      halophi = halophi > TMath::Pi() ? halophi - 2.*TMath::Pi() : halophi;
		      float haloeta = iter1->etaValue();
		      bool HaloIsGood = true;
		      // Check if halo trigger is faked by any collision muons
		      if( TheMuons.isValid() )
			{
			  float dphi = 9999.;
			  float deta = 9999.;
			  for( reco::MuonCollection::const_iterator mu = TheMuons->begin(); mu != TheMuons->end() && HaloIsGood ; mu++ )
			    {
			      // Don't match with SA-only muons
			      if( mu->isStandAloneMuon() && !mu->isTrackerMuon() && !mu->isGlobalMuon() )  continue;
			      
			      /*
			      if(!mu->isTrackerMuon())
				{
				  if( mu->isStandAloneMuon() )
				    {
				      //make sure that this SA muon is not actually a halo-like muon
				      float theta =  mu->outerTrack()->outerMomentum().theta();
				      float deta = TMath::Abs(mu->outerTrack()->outerPosition().eta() - mu->outerTrack()->innerPosition().eta());
				      if( theta < min_outer_theta || theta > max_outer_theta )  //halo-like
					continue;
				      else if ( deta > deta_threshold ) //halo-like
					continue;
				    }
				}
			      */
			    
			      const std::vector<MuonChamberMatch> chambers = mu->matches();
			      for(std::vector<MuonChamberMatch>::const_iterator iChamber = chambers.begin();
				  iChamber != chambers.end() ; iChamber ++ )
				{
				  if( iChamber->detector() != MuonSubdetId::CSC ) continue;
				  for( std::vector<reco::MuonSegmentMatch>::const_iterator iSegment = iChamber->segmentMatches.begin() ; 
				       iSegment != iChamber->segmentMatches.end(); ++iSegment )
				    {
				      edm::Ref<CSCSegmentCollection> cscSegment = iSegment->cscSegmentRef;
				      std::vector<CSCRecHit2D> hits = cscSegment -> specificRecHits();
				      for( std::vector<CSCRecHit2D>::iterator iHit = hits.begin();
					   iHit != hits.end() ; iHit++ )
					{
					  DetId TheDetUnitId(iHit->cscDetId());
					  const GeomDetUnit *TheUnit = TheCSCGeometry.idToDetUnit(TheDetUnitId);
					  LocalPoint TheLocalPosition = iHit->localPosition();
					  const BoundPlane& TheSurface = TheUnit->surface();
					  GlobalPoint TheGlobalPosition = TheSurface.toGlobal(TheLocalPosition);
					  
					  float phi_ = TheGlobalPosition.phi();
					  float eta_ = TheGlobalPosition.eta();
					  deta = deta < TMath::Abs( eta_ - haloeta ) ? deta : TMath::Abs( eta_ - haloeta );
					  dphi = dphi < TMath::Abs( phi_ - halophi ) ? dphi : TMath::Abs( phi_ - halophi );
					}
				    }
				}
			      if ( dphi < matching_dphi_threshold && deta < matching_deta_threshold) 
				HaloIsGood = false; // i.e., collision muon likely faked halo trigger
			    }
			}
		      if( !HaloIsGood ) 
			continue;
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
   short int n_alctsP=0;
   short int n_alctsM=0;
   if(TheALCTs.isValid())
     {
       for (CSCALCTDigiCollection::DigiRangeIterator j=TheALCTs->begin(); j!=TheALCTs->end(); j++) 
	 {
	   const CSCALCTDigiCollection::Range& range =(*j).second;
	   CSCDetId detId((*j).first.rawId());
	   for (CSCALCTDigiCollection::const_iterator digiIt = range.first; digiIt!=range.second; ++digiIt)
	     {
	       if( (*digiIt).isValid() && ( (*digiIt).getBX() < expected_BX ) )
		 {
		   int digi_endcap  = detId.endcap();
		   int digi_station = detId.station();
		   int digi_ring    = detId.ring();
		   int digi_chamber = detId.chamber();
		   int digi_wire    = digiIt->getKeyWG();
		   if( digi_station == 1 && digi_ring == 4 )   //hack
		     digi_ring = 1;
		   
		   bool DigiIsGood = true;
		   int dwire = 999.;
		   if( TheMuons.isValid() ) 
		     {
		       //Check if there are any collision muons with hits in the vicinity of the digi
		       for(reco::MuonCollection::const_iterator mu = TheMuons->begin(); mu!= TheMuons->end() && DigiIsGood ; mu++ )
			 {
			   if( !mu->isTrackerMuon() && !mu->isGlobalMuon() && mu->isStandAloneMuon() ) continue;
			   /*
			     if(!mu->isTrackerMuon())
				{
				  if( mu->isStandAloneMuon() )
				    {
				      //make sure that this SA muon is not actually a halo-like muon
				      float theta =  mu->outerTrack()->outerMomentum().theta();
				      float deta = TMath::Abs(mu->outerTrack()->outerPosition().eta() - mu->outerTrack()->innerPosition().eta());
				      if( theta < min_outer_theta || theta > max_outer_theta )  //halo-like
					continue;
				      else if ( deta > deta_threshold ) //halo-like
					continue;
				    }
				    }
			   */
			  
			   const std::vector<MuonChamberMatch> chambers = mu->matches();
			   for(std::vector<MuonChamberMatch>::const_iterator iChamber = chambers.begin();
			       iChamber != chambers.end(); iChamber ++ )
			     {
			       if( iChamber->detector() != MuonSubdetId::CSC ) continue;
			       for( std::vector<reco::MuonSegmentMatch>::const_iterator iSegment = iChamber->segmentMatches.begin();
				    iSegment != iChamber->segmentMatches.end(); iSegment++ )
				 {
				   edm::Ref<CSCSegmentCollection> cscSegRef = iSegment->cscSegmentRef;
				   std::vector<CSCRecHit2D> hits = cscSegRef->specificRecHits();
				   for( std::vector<CSCRecHit2D>::iterator iHit = hits.begin();
					iHit != hits.end(); iHit++ )
				     {
				       if( iHit->cscDetId().endcap() != digi_endcap ) continue;
				       if( iHit->cscDetId().station() != digi_station ) continue;
				       if( iHit->cscDetId().ring() != digi_ring ) continue;
				       if( iHit->cscDetId().chamber() != digi_chamber ) continue;
				       CSCRecHit2D::ChannelContainer hitwires = iHit->wgroups();
				       int nwires = hitwires.size();
				       int center_id = nwires/2 + 1;
				       int hit_wire = hitwires[center_id -1 ];
				       dwire = dwire < TMath::Abs(hit_wire - digi_wire)? dwire : TMath::Abs(hit_wire - digi_wire );
				     }
				 }
			     }
			   if( dwire <= matching_dwire_threshold ) 
			     DigiIsGood = false;  // collision-like muon is close to this digi
			 }
		     }
		   // only count out of time digis if they are not matched to collision muons
		   if( DigiIsGood ) 
		     {
		       if( detId.endcap() == 1 ) 
			 n_alctsP++;
		       else if ( detId.endcap() ==  2) 
			 n_alctsM++;
		     }
		 }
	     }
	 }
     }
   TheCSCHaloData.SetNOutOfTimeTriggers(n_alctsP,n_alctsM);

   // Loop over the CSCRecHit2D collection to look for out-of-time recHits
   // Out-of-time is defined as tpeak outside [t_0 + TOF - t_window, t_0 + TOF + t_window]
   // where t_0 and t_window are configurable parameters
   short int n_recHitsP = 0;
   short int n_recHitsM = 0;
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
	   float globZ = globalPosition.z();
	   if ( RHTime < (recHit_t0 - recHit_twindow) )
	     {
	       if( globZ > 0 )
		 n_recHitsP++;
	       else
		 n_recHitsM++;
	     }
	   
	   /*

	   float globX = globalPosition.x();
	   float globY = globalPosition.y();
	   float globZ = globalPosition.z();
	   float TOF = (sqrt(globX*globX+ globY*globY + globZ*globZ))/29.9792458 ; //cm -> ns
	   if ( (RHTime < (recHit_t0 + TOF - recHit_twindow)) || (RHTime > (recHit_t0 + TOF + recHit_twindow)) )
	     {
	       if( globZ > 0 ) 
		 n_recHitsP++;
	       else
		 n_recHitsM++;
	     }
	   */
	 }
     }
   TheCSCHaloData.SetNOutOfTimeHits(n_recHitsP+n_recHitsM);

   return TheCSCHaloData;
}


