
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
  recHit_t0=0.;
  recHit_twindow=25.;
  expected_BX=3;
  max_dt_muon_segment=-10.0;
  max_free_inverse_beta=0.0;
  
  min_outer_theta = 0.;
  max_outer_theta = TMath::Pi();
  
  matching_dphi_threshold = 0.18; //radians
  matching_deta_threshold = 0.4;
  matching_dwire_threshold = 5.;


  et_thresh_rh_hbhe=10; //GeV
  et_thresh_rh_ee=10; 
  et_thresh_rh_eb=10; 

  dphi_thresh_segvsrh_hbhe=0.05; //radians
  dphi_thresh_segvsrh_eb=0.05;
  dphi_thresh_segvsrh_ee=0.05; 

  dr_lowthresh_segvsrh_hbhe=-25; //cm
  dr_lowthresh_segvsrh_eb=-25; 
  dr_lowthresh_segvsrh_ee=-25;

  dr_highthresh_segvsrh_hbhe=25; //cm
  dr_highthresh_segvsrh_eb=25; 
  dr_highthresh_segvsrh_ee=25;

  dt_lowthresh_segvsrh_hbhe=0;//ns
  dt_lowthresh_segvsrh_eb=0;
  dt_lowthresh_segvsrh_ee=0;


  geo = 0;


  
}

reco::CSCHaloData CSCHaloAlgo::Calculate(const CSCGeometry& TheCSCGeometry,
					 edm::Handle<reco::MuonCollection>& TheCosmicMuons,  
					 const edm::Handle<reco::MuonTimeExtraMap> TheCSCTimeMap,
					 edm::Handle<reco::MuonCollection>& TheMuons,
					 edm::Handle<CSCSegmentCollection>& TheCSCSegments, 
					 edm::Handle<CSCRecHit2DCollection>& TheCSCRecHits,
					 edm::Handle < L1MuGMTReadoutCollection >& TheL1GMTReadout,
					 edm::Handle<HBHERecHitCollection>& hbhehits,
					 edm::Handle<EcalRecHitCollection>& ecalebhits,
					 edm::Handle<EcalRecHitCollection>& ecaleehits,
					 edm::Handle<edm::TriggerResults>& TheHLTResults,
					 const edm::TriggerNames * triggerNames, 
					 const edm::Handle<CSCALCTDigiCollection>& TheALCTs,
					 MuonSegmentMatcher *TheMatcher,  
					 const edm::Event& TheEvent,
					 const edm::EventSetup& TheSetup)
{
  reco::CSCHaloData TheCSCHaloData;
  int imucount=0;
  
  bool calomatched =false;
  if(!geo){
    edm::ESHandle<CaloGeometry> pGeo;
    TheSetup.get<CaloGeometryRecord>().get(pGeo);
    geo = pGeo.product();
  }
  bool trkmuunvetoisdefault = false; //Pb with low pt tracker muons that veto good csc segments/halo triggers. 
  //Test to "unveto" low pt trk muons. 
  //For now, we just recalculate everything without the veto and add an extra set of variables to the class CSCHaloData. 
  //If this is satisfactory, these variables can become the default ones by setting trkmuunvetoisdefault to true. 
  if( TheCosmicMuons.isValid() )
    {
      short int n_tracks_small_beta=0;
      short int n_tracks_small_dT=0;
      short int n_tracks_small_dT_and_beta=0;
      for( reco::MuonCollection::const_iterator iMuon = TheCosmicMuons->begin() ; iMuon != TheCosmicMuons->end() ; iMuon++, imucount++ )
	{
	  reco::TrackRef Track = iMuon->outerTrack();
	  if(!Track) continue;

	  bool StoreTrack = false;
	  // Calculate global phi coordinate for central most rechit in the track
	  float innermost_global_z = 1500.;
	  float outermost_global_z = 0.;
	  GlobalPoint InnerMostGlobalPosition(0.,0.,0.);  // smallest abs(z)
	  GlobalPoint OuterMostGlobalPosition(0.,0.,0.);  // largest abs(z)
	  int nCSCHits = 0;
	  for(unsigned int j = 0 ; j < Track->extra()->recHitsSize(); j++ )
	    {
	      auto hit = Track->extra()->recHitRef(j);
	      if( !hit->isValid() ) continue;
	      DetId TheDetUnitId(hit->geographicalId());
	      if( TheDetUnitId.det() != DetId::Muon ) continue;
	      if( TheDetUnitId.subdetId() != MuonSubdetId::CSC ) continue;

	      const GeomDetUnit *TheUnit = TheCSCGeometry.idToDetUnit(TheDetUnitId);
	      LocalPoint TheLocalPosition = hit->localPosition();  
	      const BoundPlane& TheSurface = TheUnit->surface();
	      const GlobalPoint TheGlobalPosition = TheSurface.toGlobal(TheLocalPosition);

	      float z = TheGlobalPosition.z();
	      if( abs(z) < innermost_global_z )
		{
		  innermost_global_z = abs(z);
		  InnerMostGlobalPosition = GlobalPoint( TheGlobalPosition);
		}
	      if( abs(z) > outermost_global_z )
		{
		  outermost_global_z = abs(z);
		  OuterMostGlobalPosition = GlobalPoint( TheGlobalPosition );
		}
	      nCSCHits ++;
	    }

	  std::vector<const CSCSegment*> MatchedSegments = TheMatcher->matchCSC(*Track,TheEvent);
	  // Find the inner and outer segments separately in case they don't agree completely with recHits
	  // Plan for the possibility segments in both endcaps
	  float InnerSegmentTime[2] = {0,0};
	  float OuterSegmentTime[2] = {0,0};
	  float innermost_seg_z[2] = {1500,1500};
	  float outermost_seg_z[2] = {0,0};
	  for (std::vector<const CSCSegment*>::const_iterator segment =MatchedSegments.begin();
	       segment != MatchedSegments.end(); ++segment)
	    {
	      CSCDetId TheCSCDetId((*segment)->cscDetId());
	      const CSCChamber* TheCSCChamber = TheCSCGeometry.chamber(TheCSCDetId);
	      LocalPoint TheLocalPosition = (*segment)->localPosition();
	      const GlobalPoint TheGlobalPosition = TheCSCChamber->toGlobal(TheLocalPosition);
	      float z = TheGlobalPosition.z();
	      int TheEndcap = TheCSCDetId.endcap();
	      if( abs(z) < innermost_seg_z[TheEndcap-1] )
		{
		  innermost_seg_z[TheEndcap-1] = abs(z);
		  InnerSegmentTime[TheEndcap-1] = (*segment)->time();
		}
	      if( abs(z) > outermost_seg_z[TheEndcap-1] )
		{
		  outermost_seg_z[TheEndcap-1] = abs(z);
		  OuterSegmentTime[TheEndcap-1] = (*segment)->time();
		}
	    }

	  if( nCSCHits < 3 ) continue; // This needs to be optimized, but is the minimum 

	  float dT_Segment = 0; // default safe value, looks like collision muon
	 
	  if( innermost_seg_z[0] < outermost_seg_z[0]) // two segments in ME+
	    dT_Segment =  OuterSegmentTime[0]-InnerSegmentTime[0];
	  if( innermost_seg_z[1] < outermost_seg_z[1]) // two segments in ME-
	    {
	      // replace the measurement if there weren't segments in ME+ or
	      // if the track in ME- has timing more consistent with an incoming particle
	      if (dT_Segment == 0.0 ||  OuterSegmentTime[1]-InnerSegmentTime[1] < dT_Segment)
		dT_Segment = OuterSegmentTime[1]-InnerSegmentTime[1] ;
	    }

	  if( OuterMostGlobalPosition.x() == 0. || OuterMostGlobalPosition.y() == 0. || OuterMostGlobalPosition.z() == 0. ) 
	    continue;
	  if( InnerMostGlobalPosition.x() == 0. || InnerMostGlobalPosition.y() == 0. || InnerMostGlobalPosition.z() == 0. )
	    continue;
	  
	  //Its a CSC Track,store it if it passes halo selection 
	  StoreTrack = true;	  

	  float deta = abs( OuterMostGlobalPosition.eta() - InnerMostGlobalPosition.eta() );
	  float dphi = abs(deltaPhi( OuterMostGlobalPosition.phi() , InnerMostGlobalPosition.phi() )) ;
	  float theta = Track->outerMomentum().theta();
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
	  if( Track->normalizedChi2() > norm_chi2_threshold )
	    StoreTrack = false;

	  if( StoreTrack )
	    {
	      TheCSCHaloData.GetCSCTrackImpactPositions().push_back( InnerMostGlobalPosition );
	      TheCSCHaloData.GetTracks().push_back( Track );
	    }

	  // Analyze the MuonTimeExtra information
	  if( TheCSCTimeMap.isValid() ) 
	    {
	      reco::MuonRef muonR(TheCosmicMuons,imucount);
	      const reco::MuonTimeExtraMap & timeMapCSC = *TheCSCTimeMap;
	      reco::MuonTimeExtra timecsc = timeMapCSC[muonR];
	      float freeInverseBeta = timecsc.freeInverseBeta();
	      
	      if (dT_Segment < max_dt_muon_segment )
		n_tracks_small_dT++;
	      if (freeInverseBeta < max_free_inverse_beta)
		n_tracks_small_beta++;
	      if ((dT_Segment < max_dt_muon_segment) &&  (freeInverseBeta < max_free_inverse_beta))
		n_tracks_small_dT_and_beta++;
	    }
	  else 
           {
	      static std::atomic<bool> MuonTimeFail{false};
              bool expected = false;
	      if( MuonTimeFail.compare_exchange_strong(expected,true,std::memory_order_acq_rel) ) 
		{
		  edm::LogWarning  ("InvalidInputTag") <<  "The MuonTimeExtraMap does not appear to be in the event. Some beam halo "
						       << " identification variables will be empty" ;
		}
	    }
	}
      TheCSCHaloData.SetNIncomingTracks(n_tracks_small_dT,n_tracks_small_beta,n_tracks_small_dT_and_beta);
    }
  else // collection is invalid
    {
      static std::atomic<bool> CosmicFail{false};
      bool expected = false;
      if( CosmicFail.compare_exchange_strong(expected,true,std::memory_order_acq_rel) ) 
	{
	  edm::LogWarning  ("InvalidInputTag") << " The Cosmic Muon collection does not appear to be in the event. These beam halo "
					       << " identification variables will be empty" ;
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
  else //  HLT results are not valid
    {
      static std::atomic<bool> HLTFail{false};
      bool expected = false;
      if( HLTFail.compare_exchange_strong(expected,true,std::memory_order_acq_rel) ) 
	{
	  edm::LogWarning  ("InvalidInputTag") << "The HLT results do not appear to be in the event. The beam halo HLT trigger "
					       << "decision will not be used in the halo identification"; 
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
       int PlusZ_alt = 0 ;
       int MinusZ_alt = 0 ;

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
		      bool HaloIsGood_alt = true;
		      // Check if halo trigger is faked by any collision muons
		      if( TheMuons.isValid() )
			{
			  float dphi = 9999.;
			  float deta = 9999.;
			  for( reco::MuonCollection::const_iterator mu = TheMuons->begin(); mu != TheMuons->end()  && (HaloIsGood ||!trkmuunvetoisdefault) ; mu++ )
			    {
			      // Don't match with SA-only muons
			      bool lowpttrackmu =false;
			      if( mu->isStandAloneMuon() && !mu->isTrackerMuon() && !mu->isGlobalMuon() )  continue;
			      if( !mu->isGlobalMuon() &&  mu->isTrackerMuon() &&  mu->pt()<3 && trkmuunvetoisdefault) continue;
			      if( !mu->isGlobalMuon() &&  mu->isTrackerMuon() &&  mu->pt()<3 ) lowpttrackmu = true;

			      /*
			      if(!mu->isTrackerMuon())
				{
				  if( mu->isStandAloneMuon() )
				    {
				      //make sure that this SA muon is not actually a halo-like muon
				      float theta =  mu->outerTrack()->outerMomentum().theta();
				      float deta = abs(mu->outerTrack()->outerPosition().eta() - mu->outerTrack()->innerPosition().eta());
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
					  
					  deta = deta < abs( eta_ - haloeta ) ? deta : abs( eta_ - haloeta );
					  dphi = dphi < abs(deltaPhi(phi_, halophi)) ? dphi : abs(deltaPhi(phi_, halophi));
					}
				    }
				}
			      if ( dphi < matching_dphi_threshold && deta < matching_deta_threshold){ 
				HaloIsGood = false; // i.e., collision muon likely faked halo trigger
				if(!lowpttrackmu)HaloIsGood_alt   = false;
			      }
			    }
			}
		      if( HaloIsGood ){ 
			if( (*iter1).etaValue() > 0 )
			  PlusZ++;
			else
			  MinusZ++;
		      }
		      if( HaloIsGood_alt ){
			if( (*iter1).etaValue() > 0 )
                          PlusZ_alt++;
                        else
                          MinusZ_alt++;
                      }

		    }
		  else
		    icsc++;
		}
	     }
	 }
       TheCSCHaloData.SetNumberOfHaloTriggers(PlusZ, MinusZ);
       TheCSCHaloData.SetNumberOfHaloTriggers_TrkMuUnVeto(PlusZ_alt, MinusZ_alt);
     }
   else
     {
       static std::atomic<bool> L1Fail{false};   
       bool expected = false;
       if( L1Fail.compare_exchange_strong(expected,true,std::memory_order_acq_rel) ) 
	 {
	   edm::LogWarning  ("InvalidInputTag") << "The L1MuGMTReadoutCollection does not appear to be in the event. The L1 beam halo trigger "
						<< "decision will not be used in the halo identification"; 
	 }
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
			   if( !mu->isGlobalMuon() &&  mu->isTrackerMuon() &&  mu->pt()<3 &&trkmuunvetoisdefault) continue;
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
				       int hit_wire = iHit->hitWire();
				       dwire = dwire < abs(hit_wire - digi_wire)? dwire : abs(hit_wire - digi_wire );
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
   else
     {
       static std::atomic<bool> DigiFail{false};
       bool expected = false;
       if (DigiFail.compare_exchange_strong(expected,true,std::memory_order_acq_rel)){
	 edm::LogWarning  ("InvalidInputTag") << "The CSCALCTDigiCollection does not appear to be in the event. The ALCT Digis will "
					      << " not be used in the halo identification"; 
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
   else
     {
       static std::atomic<bool> RecHitFail{false};
       bool expected = false;
       if( RecHitFail.compare_exchange_strong(expected,true,std::memory_order_acq_rel)  ) 
	 {
	   edm::LogWarning  ("InvalidInputTag") << "The requested CSCRecHit2DCollection does not appear to be in the event. The CSC RecHit "
						<< " variables used for halo identification will not be calculated or stored";
	 }       
     }
   TheCSCHaloData.SetNOutOfTimeHits(n_recHitsP+n_recHitsM);
   // MLR
   // Loop through CSCSegments and count the number of "flat" segments with the same (r,phi),
   // saving the value in TheCSCHaloData.
   short int maxNSegments = 0;
   bool plus_endcap = false;
   bool minus_endcap = false;
   bool both_endcaps = false;
   bool both_endcaps_loose = false;
   //   bool both_endcaps_dtcut = false;

   short int maxNSegments_alt = 0;
   bool both_endcaps_alt = false;
   bool both_endcaps_loose_alt = false;
   bool both_endcaps_loose_dtcut_alt = false;

   //float r = 0., phi = 0.;
   if (TheCSCSegments.isValid()) {
     for(CSCSegmentCollection::const_iterator iSegment = TheCSCSegments->begin();
         iSegment != TheCSCSegments->end();
         iSegment++) {

       CSCDetId iCscDetID = iSegment->cscDetId();
       bool Segment1IsGood=true;
       bool Segment1IsGood_alt=true;

       //avoid segments from collision muons
       if( TheMuons.isValid() )
	 {
	   for(reco::MuonCollection::const_iterator mu = TheMuons->begin(); mu!= TheMuons->end() && (Segment1IsGood||!trkmuunvetoisdefault)   ; mu++ )
	     {
	       bool  lowpttrackmu=false;
	       if( !mu->isTrackerMuon() && !mu->isGlobalMuon() && mu->isStandAloneMuon() ) continue;
	       if( !mu->isTrackerMuon() && !mu->isGlobalMuon() && mu->isStandAloneMuon()&&trkmuunvetoisdefault) continue;
	       if( !mu->isGlobalMuon() &&  mu->isTrackerMuon() &&  mu->pt()<3) lowpttrackmu=true;
	       const std::vector<MuonChamberMatch> chambers = mu->matches();
	       for(std::vector<MuonChamberMatch>::const_iterator kChamber = chambers.begin();
		   kChamber != chambers.end(); kChamber ++ )
		 {
		   if( kChamber->detector() != MuonSubdetId::CSC ) continue;
		   for( std::vector<reco::MuonSegmentMatch>::const_iterator kSegment = kChamber->segmentMatches.begin();
			kSegment != kChamber->segmentMatches.end(); kSegment++ )
		     {
		       edm::Ref<CSCSegmentCollection> cscSegRef = kSegment->cscSegmentRef;
		       CSCDetId kCscDetID = cscSegRef->cscDetId();
		       
		       if( kCscDetID == iCscDetID ) 
			 {
			   Segment1IsGood = false;
			   if(!lowpttrackmu) Segment1IsGood_alt=false;
			 }
		     }
		 }
	     }
	 }
       if(!Segment1IsGood&&!Segment1IsGood_alt) continue;
       
       // Get local direction vector; if direction runs parallel to beamline,
       // count this segment as beam halo candidate.
       LocalPoint iLocalPosition = iSegment->localPosition();
       LocalVector iLocalDirection = iSegment->localDirection();

       GlobalPoint iGlobalPosition = TheCSCGeometry.chamber(iCscDetID)->toGlobal(iLocalPosition);
       GlobalVector iGlobalDirection = TheCSCGeometry.chamber(iCscDetID)->toGlobal(iLocalDirection);

       float iTheta = iGlobalDirection.theta();
       if (iTheta > max_segment_theta && iTheta < TMath::Pi() - max_segment_theta) continue;
       
       float iPhi = iGlobalPosition.phi();
       float iR =  TMath::Sqrt(iGlobalPosition.x()*iGlobalPosition.x() + iGlobalPosition.y()*iGlobalPosition.y());
       float iZ = iGlobalPosition.z();
       float iT = iSegment->time();
       //       if(abs(iZ)<650&& TheEvent.id().run()< 251737) iT-= 25; 
       //Calo matching:

       bool hbhematched = HCALSegmentMatching(hbhehits,et_thresh_rh_hbhe,dphi_thresh_segvsrh_hbhe,dr_lowthresh_segvsrh_hbhe,dr_highthresh_segvsrh_hbhe,dt_lowthresh_segvsrh_hbhe,iZ,iR,iT,iPhi);
       bool ebmatched = ECALSegmentMatching(ecalebhits,et_thresh_rh_eb,dphi_thresh_segvsrh_eb,dr_lowthresh_segvsrh_eb,dr_highthresh_segvsrh_eb,dt_lowthresh_segvsrh_eb,iZ,iR,iT,iPhi);
       bool eematched = ECALSegmentMatching(ecaleehits,et_thresh_rh_ee,dphi_thresh_segvsrh_ee,dr_lowthresh_segvsrh_ee,dr_highthresh_segvsrh_ee,dt_lowthresh_segvsrh_ee,iZ,iR,iT,iPhi); 
       calomatched = calomatched? true: (hbhematched|| ebmatched|| eematched);


       short int nSegs = 0;
       short int nSegs_alt = 0;
       // Changed to loop over all Segments (so N^2) to catch as many segments as possible.
       for (CSCSegmentCollection::const_iterator jSegment = TheCSCSegments->begin();
         jSegment != TheCSCSegments->end();
         jSegment++) {
	 if (jSegment == iSegment) continue;
	 bool Segment2IsGood = true;
	 bool Segment2IsGood_alt = true;
	 LocalPoint jLocalPosition = jSegment->localPosition();
	 LocalVector jLocalDirection = jSegment->localDirection();
	 CSCDetId jCscDetID = jSegment->cscDetId();
	 GlobalPoint jGlobalPosition = TheCSCGeometry.chamber(jCscDetID)->toGlobal(jLocalPosition);
	 GlobalVector jGlobalDirection = TheCSCGeometry.chamber(jCscDetID)->toGlobal(jLocalDirection);
	 float jTheta = jGlobalDirection.theta();
	 float jPhi = jGlobalPosition.phi();
	 float jR =  TMath::Sqrt(jGlobalPosition.x()*jGlobalPosition.x() + jGlobalPosition.y()*jGlobalPosition.y());
	 float jZ = jGlobalPosition.z() ;
	 float jT = jSegment->time();
	 //	 if(abs(jZ)<650&& TheEvent.id().run() < 251737)jT-= 25;
	 if ( abs(deltaPhi(jPhi , iPhi)) <= 0.2//max_segment_phi_diff 
	     //&& abs(jR - iR) <= max_segment_r_diff 
	     && (abs(jR - iR) <= max_segment_r_diff || (abs(jR - iR)<0.03*abs(jZ - iZ) &&  jZ*iZ<0) )
	     && (jTheta < max_segment_theta || jTheta > TMath::Pi() - max_segment_theta)) {
	   //// Check if Segment matches to a colision muon
	   if( TheMuons.isValid() ) {
	     for(reco::MuonCollection::const_iterator mu = TheMuons->begin(); mu!= TheMuons->end()  && (Segment2IsGood||!trkmuunvetoisdefault) ; mu++ ) {
	       bool  lowpttrackmu=false;
	       if( !mu->isTrackerMuon() && !mu->isGlobalMuon() && mu->isStandAloneMuon() ) continue;
	       if( !mu->isGlobalMuon() &&  mu->isTrackerMuon() &&  mu->pt()<3) lowpttrackmu= true;
	       const std::vector<MuonChamberMatch> chambers = mu->matches();
	       for(std::vector<MuonChamberMatch>::const_iterator kChamber = chambers.begin();
		   kChamber != chambers.end(); kChamber ++ ) {
		 if( kChamber->detector() != MuonSubdetId::CSC ) continue;
		 for( std::vector<reco::MuonSegmentMatch>::const_iterator kSegment = kChamber->segmentMatches.begin();
		      kSegment != kChamber->segmentMatches.end(); kSegment++ ) {
		   edm::Ref<CSCSegmentCollection> cscSegRef = kSegment->cscSegmentRef;
		   CSCDetId kCscDetID = cscSegRef->cscDetId();
		   
		   if( kCscDetID == jCscDetID ) {
		     Segment2IsGood = false;
		     if(!lowpttrackmu) Segment2IsGood_alt=false;
		   }
		 }
	       }
	     }
	   }   
	   if(Segment1IsGood && Segment2IsGood) {
	     nSegs++;
	     minus_endcap = iGlobalPosition.z() < 0 || jGlobalPosition.z() < 0;
	     plus_endcap = iGlobalPosition.z() > 0 || jGlobalPosition.z() > 0;
	     //	     if( abs(jT-iT)/sqrt( (jR-iR)*(jR-iR)+(jZ-iZ)*(jZ-iZ) )<0.05 && abs(jT-iT)/sqrt( (jR-iR)*(jR-iR)+(jZ-iZ)*(jZ-iZ) )>0.02 && minus_endcap&&plus_endcap ) both_endcaps_dtcut =true;
	   }
	   if(Segment1IsGood_alt && Segment2IsGood_alt) {
             nSegs_alt++;
             minus_endcap = iGlobalPosition.z() < 0 || jGlobalPosition.z() < 0;
             plus_endcap = iGlobalPosition.z() > 0 || jGlobalPosition.z() > 0;
             if( abs(jT-iT)<0.05*sqrt( (jR-iR)*(jR-iR)+(jZ-iZ)*(jZ-iZ) ) && abs(jT-iT)> 0.02*sqrt( (jR-iR)*(jR-iR)+(jZ-iZ)*(jZ-iZ) ) && minus_endcap&&plus_endcap ) both_endcaps_loose_dtcut_alt =true;
           }
	   
	 }
       }
       // Correct the fact that the way nSegs counts will always be short by 1
       if (nSegs > 0) nSegs++;
       
       // The opposite endcaps segments do not need to belong to the longest chain. 
       if (nSegs > 0) both_endcaps_loose =  both_endcaps_loose ? both_endcaps_loose : minus_endcap && plus_endcap;
       if (nSegs_alt > 0) nSegs_alt++;
       if (nSegs_alt > 0) both_endcaps_loose_alt =  both_endcaps_loose_alt ? both_endcaps_loose_alt : minus_endcap && plus_endcap;

       //       if (nSegs > 0) both_endcaps_dt20ns = both_endcaps_dt20ns ? both_endcaps_dt20ns : minus_endcap && plus_endcap &&dt20ns;
       if (nSegs > maxNSegments) {
	 // Use value of r, phi to collect halo CSCSegments for examining timing (not coded yet...)
	 //r = iR;
	 //phi = iPhi;
	 maxNSegments = nSegs;
	 both_endcaps = both_endcaps ? both_endcaps : minus_endcap && plus_endcap;
       }
      
       if (nSegs_alt > maxNSegments_alt) {
       	 maxNSegments_alt = nSegs_alt;
         both_endcaps_alt = both_endcaps_alt ? both_endcaps_alt : minus_endcap && plus_endcap;
       }
 
     }
   }
   TheCSCHaloData.SetNFlatHaloSegments(maxNSegments);
   TheCSCHaloData.SetSegmentsBothEndcaps(both_endcaps);
   TheCSCHaloData.SetNFlatHaloSegments_TrkMuUnVeto(maxNSegments_alt);
   TheCSCHaloData.SetSegmentsBothEndcaps_Loose_TrkMuUnVeto(both_endcaps_loose_alt);
   TheCSCHaloData.SetSegmentsBothEndcaps_Loose_dTcut_TrkMuUnVeto(both_endcaps_loose_dtcut_alt);
   TheCSCHaloData.SetSegmentIsCaloMatched(calomatched);

   return TheCSCHaloData;
}

math::XYZPoint CSCHaloAlgo::getPosition(const DetId &id, reco::Vertex::Point vtx){

  const GlobalPoint& pos=geo->getPosition(id);
  math::XYZPoint posV(pos.x() - vtx.x(),pos.y() - vtx.y(),pos.z() - vtx.z());
  return posV;
}


bool CSCHaloAlgo::HCALSegmentMatching(edm::Handle<HBHERecHitCollection>& rechitcoll, float et_thresh_rh, float dphi_thresh_segvsrh, float dr_lowthresh_segvsrh, float dr_highthresh_segvsrh, float dt_lowthresh_segvsrh , float iZ, float iR, float iT, float iPhi){
  reco::Vertex::Point vtx(0,0,0);
  for(size_t ihit = 0; ihit< rechitcoll->size(); ++ ihit){
    const HBHERecHit & rechit = (*rechitcoll)[ ihit ];
    math::XYZPoint rhpos = getPosition(rechit.id(),vtx);
    double rhet = rechit.energy()/cosh(rhpos.eta());
    double dphi_rhseg = abs(deltaPhi(rhpos.phi(),iPhi));
    double dr_rhseg = sqrt(rhpos.x()*rhpos.x()+rhpos.y()*rhpos.y()) - iR;
    double dtcorr_rhseg = rechit.time()- abs(rhpos.z()-iZ)/30- iT; 
    if(rhet> et_thresh_rh&&
       dphi_rhseg < dphi_thresh_segvsrh &&
       dr_rhseg < dr_highthresh_segvsrh && dr_rhseg> dr_lowthresh_segvsrh && //careful: asymmetric cut might not be the most appropriate thing 
       dtcorr_rhseg> dt_lowthresh_segvsrh
      ) return true; 
  }
  return false;
}

bool CSCHaloAlgo::ECALSegmentMatching(edm::Handle<EcalRecHitCollection>& rechitcoll,  float et_thresh_rh, float dphi_thresh_segvsrh, float dr_lowthresh_segvsrh, float dr_highthresh_segvsrh, float dt_lowthresh_segvsrh, float iZ, float iR, float iT, float iPhi ){
  reco::Vertex::Point vtx(0,0,0);
  for(size_t ihit = 0; ihit<rechitcoll->size(); ++ ihit){
    const EcalRecHit & rechit = (*rechitcoll)[ ihit ];
    math::XYZPoint rhpos = getPosition(rechit.id(),vtx);
    double rhet = rechit.energy()/cosh(rhpos.eta());
    double dphi_rhseg = abs(deltaPhi(rhpos.phi(),iPhi));
    double dr_rhseg = sqrt(rhpos.x()*rhpos.x()+rhpos.y()*rhpos.y()) - iR;
    double dtcorr_rhseg = rechit.time()- abs(rhpos.z()-iZ)/30- iT; 
    if(rhet> et_thresh_rh&&
       dphi_rhseg < dphi_thresh_segvsrh &&
       dr_rhseg < dr_highthresh_segvsrh && dr_rhseg> dr_lowthresh_segvsrh && //careful: asymmetric cut might not be the most appropriate thing 
       dtcorr_rhseg> dt_lowthresh_segvsrh
       ) return true; 
  }
  return false;
}
