#include "RecoMET/METAlgorithms/interface/CSCHaloAlgo.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "RecoMET/METAlgorithms/interface/HaloClusterCandidateEB.h"
#include "RecoMET/METAlgorithms/interface/HaloClusterCandidateEE.h"
#include "RecoMET/METAlgorithms/interface/HaloClusterCandidateHB.h"
#include "RecoMET/METAlgorithms/interface/HaloClusterCandidateHE.h"
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


  et_thresh_rh_hbhe=25;//to be removed
  dphi_thresh_segvsrh_hbhe=0.05;
  dr_lowthresh_segvsrh_hbhe=-100;
  dr_highthresh_segvsrh_hbhe=20;
  dt_highthresh_segvsrh_hbhe=30;
  

  et_thresh_rh_hb=20; //GeV
  et_thresh_rh_he=20;
  et_thresh_rh_ee=10; 
  et_thresh_rh_eb=10; 

  dphi_thresh_segvsrh_hb=0.15; //radians
  dphi_thresh_segvsrh_he=0.1;
  dphi_thresh_segvsrh_eb=0.04;
  dphi_thresh_segvsrh_ee=0.04; 

  dr_lowthresh_segvsrh_hb=-100; //cm
  dr_lowthresh_segvsrh_he=-30; 
  dr_lowthresh_segvsrh_eb=-30; 
  dr_lowthresh_segvsrh_ee=-30;

  dr_highthresh_segvsrh_hb=20; //cm
  dr_highthresh_segvsrh_he=30;
  dr_highthresh_segvsrh_eb=15; 
  dr_highthresh_segvsrh_ee=30;

  dt_highthresh_segvsrh_hb=15;//ns
  dt_highthresh_segvsrh_he=15;
  dt_highthresh_segvsrh_eb=15;
  dt_highthresh_segvsrh_ee=15;

  dt_segvsrh_hb=15;//ns
  dt_segvsrh_he=15;
  dt_segvsrh_eb=15;
  dt_segvsrh_ee=15;
  

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
					 const edm::EventSetup& TheSetup,
					 bool ishlt)
{

  reco::CSCHaloData TheCSCHaloData;
  int imucount=0;
  std::auto_ptr<EcalRecHitCollection> rechitsEB_bhmatched(new EcalRecHitCollection());
  bool calomatched =false;
  bool ECALBmatched =false;
  bool ECALEmatched =false;
  bool HCALBmatched =false;
  bool HCALEmatched =false;


  geo = 0;
  edm::ESHandle<CaloGeometry> pGeo;
  TheSetup.get<CaloGeometryRecord>().get(pGeo);
  geo = pGeo.product();

  bool trkmuunvetoisdefault = false; //Pb with low pt tracker muons that veto good csc segments/halo triggers. 
  //Test to "unveto" low pt trk muons. 
  //For now, we just recalculate everything without the veto and add an extra set of variables to the class CSCHaloData. 
  //If this is satisfactory, these variables can become the default ones by setting trkmuunvetoisdefault to true. 



  //Halo cluster building:
  //Various clusters are built, depending on the subdetector.
  //In barrel, one looks for deposits narrow in phi.
  //In endcaps, one looks for localized deposits (dr condition in EE where r =sqrt(dphi*dphi+deta*deta), and  dR and dphi conditions in HE, where dR = difference of radial coordinates between the hits) . 
  //H/E or E/H conditions are also applied for EB, HB, HE.
  //
  //The halo cluster building step targets an efficiency of 99% for beam halo deposits. 
  //
  //These clusters are used as input for the halo pattern finding methods and for the CSC-calo matching methods. 
  std::vector<HaloClusterCandidateEB> haloclustercands_EB;
  haloclustercands_EB=  GetHaloClusterCandidateEB(ecalebhits , hbhehits, 5);
  std::vector<HaloClusterCandidateEE> haloclustercands_EE;
  haloclustercands_EE=  GetHaloClusterCandidateEE(ecaleehits , hbhehits, 10);
  std::vector<HaloClusterCandidateHB> haloclustercands_HB;
  haloclustercands_HB=  GetHaloClusterCandidateHB(ecalebhits , hbhehits, 10);
  std::vector<HaloClusterCandidateHE> haloclustercands_HE;
  haloclustercands_HE=  GetHaloClusterCandidateHE(ecaleehits , hbhehits, 20);
  

  //Halo pattern finding: 
  //In barrel, one looks for deposits spread along many rechits with constant phi and/or containing a lot of OOT rechits. 
  //In EE, one looks for deposits containing OOT rechits. 
  //In HE, one lookst for deposits fully contained in one HCAL layer.
  bool HaloPatternFoundInEB = EBClusterShapeandTimeStudy(TheCSCHaloData,haloclustercands_EB,ishlt); 
  bool HaloPatternFoundInEE = EEClusterShapeandTimeStudy(TheCSCHaloData,haloclustercands_EE,ishlt); 
  bool HaloPatternFoundInHB = HBClusterShapeandTimeStudy(TheCSCHaloData,haloclustercands_HB,ishlt); 
  bool HaloPatternFoundInHE = HEClusterShapeandTimeStudy(TheCSCHaloData,haloclustercands_HE,ishlt); 




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
	  float dphi = abs(deltaPhi( OuterMostGlobalPosition.barePhi() , InnerMostGlobalPosition.barePhi() )) ;
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
       
       //CSC-calo matching: 
       //Here, one checks if any halo cluster can be matched to a CSC segment. 
       //The matching uses both geometric (dphi, dR) and timing information (dt).
       //The cut values depend on the subdetector considered (e.g. in HB, Rcalo-Rsegment is allowed to be very negative) 

       bool ebmatched =SegmentMatchingEB(TheCSCHaloData,haloclustercands_EB,iZ,iR,iT,iPhi,ishlt);
       bool eematched =SegmentMatchingEE(TheCSCHaloData,haloclustercands_EE,iZ,iR,iT,iPhi,ishlt);
       bool hbmatched =SegmentMatchingHB(TheCSCHaloData,haloclustercands_HB,iZ,iR,iT,iPhi,ishlt);
       bool hematched =SegmentMatchingHE(TheCSCHaloData,haloclustercands_HE,iZ,iR,iT,iPhi,ishlt);
       
       calomatched = calomatched? true : ( ebmatched|| eematched || hbmatched||hematched);
       ECALBmatched = ECALBmatched? true :ebmatched;
       ECALEmatched = ECALEmatched? true :eematched;
       HCALBmatched = HCALBmatched? true :hbmatched;
       HCALEmatched = HCALEmatched? true :hematched;

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

	 if ( abs(deltaPhi(jPhi , iPhi)) <=max_segment_phi_diff 
	      && ( abs(jR - iR) <0.05*abs(jZ - iZ)||  jZ*iZ>0 )
	      && ( (jR - iR) >-0.02*abs(jZ - iZ) || iT>jT ||  jZ*iZ>0)
	      && ( (iR - jR) >-0.02*abs(jZ - iZ) || iT<jT ||  jZ*iZ>0)
	      && (abs(jR - iR) <= max_segment_r_diff ||  jZ*iZ < 0)
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
	   }
	   if(Segment1IsGood_alt && Segment2IsGood_alt) {
             nSegs_alt++;
             minus_endcap = iGlobalPosition.z() < 0 || jGlobalPosition.z() < 0;
             plus_endcap = iGlobalPosition.z() > 0 || jGlobalPosition.z() > 0;
             if( 
		abs(jT-iT)<0.05*sqrt( (jR-iR)*(jR-iR)+(jZ-iZ)*(jZ-iZ) ) && 
		abs(jT-iT)> 0.02*sqrt( (jR-iR)*(jR-iR)+(jZ-iZ)*(jZ-iZ) ) && 
		(iT>-15 || jT>-15)&&
		minus_endcap&&plus_endcap ) both_endcaps_loose_dtcut_alt =true;
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
   
   //Deprecated methods, kept for backward compatibility
   TheCSCHaloData.SetHLTBit(false);
   TheCSCHaloData.SetNumberOfHaloTriggers(0,0);
   TheCSCHaloData.SetNumberOfHaloTriggers_TrkMuUnVeto(0,0);
   TheCSCHaloData.SetNOutOfTimeTriggers(0,0);

   //Current methods used
   TheCSCHaloData.SetNFlatHaloSegments(maxNSegments);
   TheCSCHaloData.SetSegmentsBothEndcaps(both_endcaps);
   TheCSCHaloData.SetNFlatHaloSegments_TrkMuUnVeto(maxNSegments_alt);
   TheCSCHaloData.SetSegmentsBothEndcaps_Loose_TrkMuUnVeto(both_endcaps_loose_alt);
   TheCSCHaloData.SetSegmentsBothEndcaps_Loose_dTcut_TrkMuUnVeto(both_endcaps_loose_dtcut_alt);
   TheCSCHaloData.SetSegmentIsCaloMatched(calomatched);
   TheCSCHaloData.SetSegmentIsEBCaloMatched(ECALBmatched);
   TheCSCHaloData.SetSegmentIsEECaloMatched(ECALEmatched);
   TheCSCHaloData.SetSegmentIsHBCaloMatched(HCALBmatched);
   TheCSCHaloData.SetSegmentIsHECaloMatched(HCALEmatched);
   TheCSCHaloData.SetHaloPatternFoundEB(HaloPatternFoundInEB);
   TheCSCHaloData.SetHaloPatternFoundEE(HaloPatternFoundInEE);
   TheCSCHaloData.SetHaloPatternFoundHB(HaloPatternFoundInHB);
   TheCSCHaloData.SetHaloPatternFoundHE(HaloPatternFoundInHE);

   /*
   //To be removed, just for tests
   reco::Vertex::Point vtx(0,0,0);
   for(size_t jhit = 0; jhit<hbhehits->size(); ++ jhit){
     const HBHERecHit & rechitj = (*hbhehits)[ jhit ];
     math::XYZPoint rhposj = getPosition(rechitj.id(),vtx);
     double rhetj = rechitj.energy()/cosh(rhposj.eta());
     if(rhetj<2) continue;
     cout << "HBHErhet: "<< rhetj
	  << " rheta " << rhposj.eta()
	  << " rhphi " << rhposj.phi()
	  << " rhtime: " << rechitj.time()
	  << " rhZ: " << rhposj.z()
	  << " rhR: " << sqrt(rhposj.x()*rhposj.x()+rhposj.y()*rhposj.y())
	  << endl; 
   }
   cout << endl;
   for(size_t jhit = 0; jhit<ecalebhits->size(); ++ jhit){
     const EcalRecHit & rechitj = (*ecalebhits)[ jhit ];
     math::XYZPoint rhposj = getPosition(rechitj.id(),vtx);
     double rhetj = rechitj.energy()/cosh(rhposj.eta());
     if(rhetj<2) continue;
     cout << "EBrhet: "<< rhetj
	  << " rheta " << rhposj.eta()
	  << " rhphi " << rhposj.phi()
	  << " rhtime: " << rechitj.time()
	  << " rhZ: " << rhposj.z()
	  << " rhR: " << sqrt(rhposj.x()*rhposj.x()+rhposj.y()*rhposj.y())
	  << endl; 
     
   }
   cout << endl;
  for(size_t jhit = 0; jhit<ecaleehits->size(); ++ jhit){
    const EcalRecHit & rechitj = (*ecaleehits)[ jhit ];
    math::XYZPoint rhposj = getPosition(rechitj.id(),vtx);
    double rhetj = rechitj.energy()/cosh(rhposj.eta());
    if(rhetj<2) continue;
    cout << "EErhet: "<< rhetj
	 << " rheta " << rhposj.eta()
	 << " rhphi " << rhposj.phi()
	 << " rhtime: " << rechitj.time()
	 << " rhZ: " << rhposj.z()
	 << " rhR: " << sqrt(rhposj.x()*rhposj.x()+rhposj.y()*rhposj.y())
	 << endl; 
  }


   */


   return TheCSCHaloData;
}

math::XYZPoint CSCHaloAlgo::getPosition(const DetId &id, reco::Vertex::Point vtx){

  const GlobalPoint& pos=geo->getPosition(id);
  math::XYZPoint posV(pos.x() - vtx.x(),pos.y() - vtx.y(),pos.z() - vtx.z());
  return posV;
}


bool CSCHaloAlgo::EBClusterShapeandTimeStudy(reco::CSCHaloData & thehalodata,std::vector<HaloClusterCandidateEB>haloclustercands, bool ishlt){
  //Conditions on the central strip size in eta.
  //For low size, extra conditions on seed et, isolation and cluster timing 
  //The time condition only targets IT beam halo. 
  //EB rechits from OT beam halos are typically too late (around 5 ns or more) and seem therefore already cleaned by the reconstruction.
  bool halofound = false;
  for(size_t i = 0; i <haloclustercands.size(); i++){
      if(haloclustercands[i].GetSeedEt()<5)continue;
      if(haloclustercands[i].GetNbofCrystalsInEta()<4) continue;
      if(haloclustercands[i].GetNbofCrystalsInEta()==4&&haloclustercands[i].GetSeedEt()<10) continue;
      if(haloclustercands[i].GetNbofCrystalsInEta()==4 && haloclustercands[i].GetEtStripIPhiSeedPlus1()>0.1 &&haloclustercands[i].GetEtStripIPhiSeedMinus1()>0.1 ) continue;
      if(haloclustercands[i].GetNbofCrystalsInEta()<=5 &&  haloclustercands[i].GetTimeDiscriminator()>=0.)continue; 
      
      //For HLT, only use conditions without timing and tighten seed et condition
      if(ishlt &&haloclustercands[i].GetNbofCrystalsInEta()<=5)continue;
      if(ishlt && haloclustercands[i].GetSeedEt()<10)continue;
      halofound =true;
      
      edm::RefVector<EcalRecHitCollection> bhrhcandidates = haloclustercands[i].GetBeamHaloRecHitsCandidates();
      if(halofound)AddtoBeamHaloEBEERechits(bhrhcandidates, thehalodata,true);
      
    }
    return halofound;
}



bool CSCHaloAlgo::EEClusterShapeandTimeStudy(reco::CSCHaloData & thehalodata,std::vector<HaloClusterCandidateEE>haloclustercands, bool ishlt){
  //Separate conditions targeting IT and OT beam halos
  bool halofound = false;

  //For OT beam halos, just require enough crystals with large T
    for(size_t i = 0; i <haloclustercands.size(); i++){
      if(haloclustercands[i].GetSeedEt()<20)continue;
      if(haloclustercands[i].GetSeedTime()<0.5)continue;
      if(haloclustercands[i].GetNbLateCrystals()-haloclustercands[i].GetNbEarlyCrystals() <2)continue;
      
      //The use of time information does not allow this method to work at HLT
      if(ishlt)continue;
      halofound =true;
      edm::RefVector<EcalRecHitCollection> bhrhcandidates = haloclustercands[i].GetBeamHaloRecHitsCandidates();
      if(halofound)AddtoBeamHaloEBEERechits(bhrhcandidates, thehalodata,false);
      
    }

    //For IT beam halos, fakes from collisions are higher => require the cluster size to be small. 
    //Only halos with R>100 cm are considered here.
    //For lower values, the time difference with particles from collisions is too small
    //IT outgoing beam halos that interact in EE at low R is probably the most difficult category to deal with: 
    //Their signature is very close to the one of photon from collisions (similar cluster shape and timing)
    for(size_t i = 0; i <haloclustercands.size(); i++){
      if(haloclustercands[i].GetSeedEt()<20)continue;
      if(haloclustercands[i].GetSeedR()<100)continue;
      if(haloclustercands[i].GetTimeDiscriminator()<1) continue; 
      if(haloclustercands[i].GetClusterSize()<2) continue;
      if(haloclustercands[i].GetClusterSize()>4) continue;
      
      //The use of time information does not allow this method to work at HLT
      if(ishlt)continue;
      halofound =true;
      edm::RefVector<EcalRecHitCollection> bhrhcandidates = haloclustercands[i].GetBeamHaloRecHitsCandidates();
      if(halofound) AddtoBeamHaloEBEERechits(bhrhcandidates, thehalodata,false);


    }
    
    return halofound;

}



bool CSCHaloAlgo::HBClusterShapeandTimeStudy(reco::CSCHaloData & thehalodata,std::vector<HaloClusterCandidateHB>haloclustercands, bool ishlt){
  //Conditions on the central strip size in eta.
  //For low size, extra conditions on seed et, isolation and cluster timing 
  //Here we target both IT and OT beam halo. Two separate discriminators were built for the two cases.
  bool halofound = false;
  for(size_t i = 0; i <haloclustercands.size(); i++){
    if(haloclustercands[i].GetSeedEt()<10)continue;
    if(haloclustercands[i].GetNbTowersInEta()<3) continue;
    //Isolation criteria for very short eta strips
    if(haloclustercands[i].GetNbTowersInEta()==3 && (haloclustercands[i].GetEtStripPhiSeedPlus1()>0.1 || haloclustercands[i].GetEtStripPhiSeedMinus1()>0.1) ) continue;
    if(haloclustercands[i].GetNbTowersInEta()<=5 && (haloclustercands[i].GetEtStripPhiSeedPlus1()>0.1 && haloclustercands[i].GetEtStripPhiSeedMinus1()>0.1) ) continue;
    //Timing conditions for short eta strips
    if(haloclustercands[i].GetNbTowersInEta()==3 && haloclustercands[i].GetTimeDiscriminatorITBH()>=0.) continue;
    if(haloclustercands[i].GetNbTowersInEta()<=6 && haloclustercands[i].GetTimeDiscriminatorITBH()>=5. &&haloclustercands[i].GetTimeDiscriminatorOTBH()<0.) continue; 
    
    //For HLT, only use conditions without timing 
    if(ishlt && haloclustercands[i].GetNbTowersInEta()<7) continue;
    halofound =true;
    
    edm::RefVector<HBHERecHitCollection> bhrhcandidates = haloclustercands[i].GetBeamHaloRecHitsCandidates();
    if(halofound)AddtoBeamHaloHBHERechits(bhrhcandidates, thehalodata);
    
  }
  return halofound;
}

bool CSCHaloAlgo::HEClusterShapeandTimeStudy(reco::CSCHaloData & thehalodata,std::vector<HaloClusterCandidateHE>haloclustercands, bool ishlt){
  //Conditions on H1/H123 to spot halo interacting only in one HCAL layer. 
  //For R> about 170cm, HE has only one layer and this condition cannot be applied
  //Note that for R>170 cm, the halo is in CSC acceptance and will most likely be spotted by the CSC-calo matching method
  //A method to identify halos interacting in both H1 and H2/H3 at low R is still missing. 
  bool halofound = false;
 
    for(size_t i = 0; i <haloclustercands.size(); i++){
      if(haloclustercands[i].GetSeedEt()<20)continue;
      if(haloclustercands[i].GetSeedR()>170) continue;
      if(haloclustercands[i].GetH1overH123()>0.02 &&haloclustercands[i].GetH1overH123()<0.98) continue;
      
      //This method is one of the ones with the highest fake rate: in JetHT dataset, it happens in around 0.1% of the cases that a low pt jet (pt= 20) leaves all of its energy in only one HCAL layer. 
      //At HLT, one only cares about large deposits from BH that would lead to a MET/SinglePhoton trigger to be fired.
      //Rising the seed Et threshold at HLT has therefore little impact on the HLT performances but ensures that possible controversial events are still recorded.
      if(ishlt && haloclustercands[i].GetSeedEt()<50)continue;
      halofound =true;
      edm::RefVector<HBHERecHitCollection> bhrhcandidates = haloclustercands[i].GetBeamHaloRecHitsCandidates();
      if(halofound)AddtoBeamHaloHBHERechits(bhrhcandidates, thehalodata);
      
    }
    return halofound;
}


void CSCHaloAlgo::AddtoBeamHaloHBHERechits(edm::RefVector<HBHERecHitCollection>& bhtaggedrechits,reco::CSCHaloData & thehalodata){  
  for(size_t ihit = 0; ihit<bhtaggedrechits.size(); ++ ihit){
    bool alreadyincl = false; 
    edm::Ref<HBHERecHitCollection> rhRef( bhtaggedrechits[ihit] ) ;
    edm::RefVector<HBHERecHitCollection> refrhcoll;
    refrhcoll=thehalodata.GetHBHERechits();
    for(size_t jhit =0; jhit < refrhcoll.size();jhit++){                                                                                                                                                 
      edm::Ref<HBHERecHitCollection> rhitRef( refrhcoll[jhit] ) ;                                                                                                                                        
      if(rhitRef->detid() == rhRef->detid()) alreadyincl=true;                                                                                                                                           
      if(rhitRef->detid() == rhRef->detid()) break;                                                                                                                                                      
    } 
    if(!alreadyincl)thehalodata.GetHBHERechits().push_back(rhRef);
  }
}
 
void CSCHaloAlgo::AddtoBeamHaloEBEERechits(edm::RefVector<EcalRecHitCollection>& bhtaggedrechits,reco::CSCHaloData & thehalodata, bool isbarrel){
  for(size_t ihit = 0; ihit<bhtaggedrechits.size(); ++ ihit){
    bool alreadyincl = false; 
    edm::Ref<EcalRecHitCollection> rhRef( bhtaggedrechits[ihit] ) ;
    edm::RefVector<EcalRecHitCollection> refrhcoll;
    if(isbarrel) refrhcoll=thehalodata.GetEBRechits();
    else refrhcoll=thehalodata.GetEERechits();
    for(size_t jhit =0; jhit < refrhcoll.size();jhit++){                                                                                                                                                 
      edm::Ref<EcalRecHitCollection> rhitRef( refrhcoll[jhit] ) ;                                                                                                                                        
      if(rhitRef->detid() == rhRef->detid()) alreadyincl=true;                                                                                                                                           
      if(rhitRef->detid() == rhRef->detid()) break;                                                                                                                                                      
    } 
    if(!alreadyincl&&isbarrel)thehalodata.GetEBRechits().push_back(rhRef);
    if(!alreadyincl&&!isbarrel)thehalodata.GetEERechits().push_back(rhRef);
  }  
}
 
 
 
std::vector<HaloClusterCandidateEB> CSCHaloAlgo::GetHaloClusterCandidateEB(edm::Handle<EcalRecHitCollection>& ecalrechitcoll, edm::Handle<HBHERecHitCollection>& hbherechitcoll,float et_thresh_seedrh){

  std::vector<HaloClusterCandidateEB> TheHaloClusterCandsEB;
  reco::Vertex::Point vtx(0,0,0);

  for(size_t ihit = 0; ihit<ecalrechitcoll->size(); ++ ihit){
    HaloClusterCandidateEB  clustercand;
    
    const EcalRecHit & rechit = (*ecalrechitcoll)[ ihit ];
    math::XYZPoint rhpos = getPosition(rechit.id(),vtx);
    //Et condition
    double rhet = rechit.energy()/cosh(rhpos.eta());
    if(rhet<et_thresh_seedrh) continue;
    double eta = rhpos.eta();
    double phi = rhpos.phi();
    
    bool isiso = true;
    double etcluster(0);
    int nbcrystalsameeta(0);
    double timediscriminator(0);
    double etstrip_iphiseedplus1(0), etstrip_iphiseedminus1(0);

    //Building the cluster 
    edm::RefVector<EcalRecHitCollection> bhrhcandidates;
    for(size_t jhit = 0; jhit<ecalrechitcoll->size(); ++ jhit){
      const EcalRecHit & rechitj = (*ecalrechitcoll)[ jhit ];
      EcalRecHitRef rhRef(ecalrechitcoll,jhit);
      math::XYZPoint rhposj = getPosition(rechitj.id(),vtx);

      double etaj = rhposj.eta();
      double phij = rhposj.phi();
      if(abs(eta-etaj)>0.2) continue;//This means +/-11 crystals in eta 
      if(abs(deltaPhi(phi,phij))>0.08) continue;//This means +/-4 crystals in phi 
      
      double rhetj = rechitj.energy()/cosh(rhposj.eta());
      //Rechits with et between 1 and 2 GeV are saved in the rh list but not used in the calculation of the halocluster variables
      if(rhetj<1) continue;
      bhrhcandidates.push_back(rhRef);
      if(rhetj<2) continue;

      if(abs(deltaPhi(phi,phij))>0.03){isiso=false;break;}//The strip should be isolated
      if(abs(deltaPhi(phi,phij))<0.01) nbcrystalsameeta++;
      if(deltaPhi(phi,phij)>0.01) etstrip_iphiseedplus1+=rhetj;
      if(deltaPhi(phi,phij)<-0.01) etstrip_iphiseedminus1+=rhetj;
      etcluster+=rhetj;
      //Timing discriminator
      //We assign a weight to the rechit defined as: 
      //Log10(Et)*f(T,R,Z) 
      //where f(T,R,Z) is the separation curve between halo-like and IP-like times.
      //The time difference between a deposit from a outgoing IT halo and a deposit coming from a particle emitted at the IP is given by: 
      //dt= ( - sqrt(R^2+z^2) + |z| )/c
      //Here we take R to be 130 cm. 
      //For EB, the function was parametrized as a function of ieta instead of Z.
      double rhtj = rechitj.time();
      EBDetId detj    = rechitj.id();
      int rhietaj= detj.ieta();
      timediscriminator+= TMath::Log10( rhetj )* ( rhtj +0.5*(sqrt(16900+9*rhietaj*rhietaj)-3*abs(rhietaj))/30 );
      
    }
    //Isolation condition
    if(!isiso) continue;
    
    //Calculate H/E
    double hoe(0);
    for(size_t jhit = 0; jhit<hbherechitcoll->size(); ++ jhit){
      const HBHERecHit & rechitj = (*hbherechitcoll)[ jhit ];
      math::XYZPoint rhposj = getPosition(rechitj.id(),vtx);
      double rhetj = rechitj.energy()/cosh(rhposj.eta());
      if(rhetj<2) continue;
      double etaj = rhposj.eta();
      double phij = rhposj.phi();
      if(abs(eta-etaj)>0.2) continue;
      if(abs(deltaPhi(phi,phij))>0.2) continue;
      hoe+=rhetj/etcluster;
    }
    //H/E condition
    if(hoe>0.1) continue; 
        

    clustercand.SetClusterEt(etcluster); 
    clustercand.SetSeedEt(rhet); 
    clustercand.SetSeedEta(eta); 
    clustercand.SetSeedPhi(phi); 
    clustercand.SetSeedZ(rhpos.Z());
    clustercand.SetSeedR(sqrt(rhpos.x()*rhpos.x()+rhpos.y()*rhpos.y())); 
    clustercand.SetSeedTime(rechit.time()); 
    clustercand.SetHoverE(hoe);
    clustercand.SetNbofCrystalsInEta(nbcrystalsameeta);
    clustercand.SetEtStripIPhiSeedPlus1(etstrip_iphiseedplus1);
    clustercand.SetEtStripIPhiSeedMinus1(etstrip_iphiseedminus1);
    clustercand.SetTimeDiscriminator(timediscriminator);
    clustercand.SetBeamHaloRecHitsCandidates(bhrhcandidates);

    TheHaloClusterCandsEB.push_back(clustercand);
  }

  return  TheHaloClusterCandsEB;
} 




std::vector<HaloClusterCandidateEE> CSCHaloAlgo::GetHaloClusterCandidateEE(edm::Handle<EcalRecHitCollection>& ecalrechitcoll, edm::Handle<HBHERecHitCollection>& hbherechitcoll,float et_thresh_seedrh){

  std::vector<HaloClusterCandidateEE> TheHaloClusterCandsEE;

  reco::Vertex::Point vtx(0,0,0);

  for(size_t ihit = 0; ihit<ecalrechitcoll->size(); ++ ihit){
    HaloClusterCandidateEE  clustercand;
    
    const EcalRecHit & rechit = (*ecalrechitcoll)[ ihit ];
    math::XYZPoint rhpos = getPosition(rechit.id(),vtx);
    //Et condition
    double rhet = rechit.energy()/cosh(rhpos.eta());
    if(rhet<et_thresh_seedrh) continue;
    double eta = rhpos.eta();
    double phi = rhpos.phi();
    double rhr = sqrt(rhpos.x()*rhpos.x()+rhpos.y()*rhpos.y());
    
    bool isiso = true;
    double etcluster(0);
    double timediscriminator(0);
    int clustersize(0);
    int nbcrystalssmallt(0);
    int nbcrystalshight(0);
    //Building the cluster
    edm::RefVector<EcalRecHitCollection> bhrhcandidates;
    for(size_t jhit = 0; jhit<ecalrechitcoll->size(); ++ jhit){
      const EcalRecHit & rechitj = (*ecalrechitcoll)[ jhit ];
      EcalRecHitRef rhRef(ecalrechitcoll,jhit);
      math::XYZPoint rhposj = getPosition(rechitj.id(),vtx);

      //Ask the hits to be in the same endcap
      if(rhposj.z()*rhpos.z()<0)continue;

      double etaj = rhposj.eta();
      double phij = rhposj.phi();
      double dr = sqrt(abs(eta-etaj)*abs(eta-etaj)+deltaPhi(phi,phij)*deltaPhi(phi,phij));

      //Outer cone
      if(dr>0.3) continue;
      
      double rhetj = rechitj.energy()/cosh(rhposj.eta());
      //Rechits with et between 1 and 2 GeV are saved in the rh list but not used in the calculation of the halocluster variables
      if(rhetj<1) continue; 
      bhrhcandidates.push_back(rhRef);
      if(rhetj<2) continue;
      
      //Isolation between outer and inner cone
      if(dr>0.05){isiso=false;break;}//The deposit should be isolated

      etcluster+=rhetj;
  
      //Timing infos:
      //Here we target both IT and OT beam halo 
      double rhtj=rechitj.time();

      //Discriminating variables for OT beam halo: 
      if(rhtj>1) nbcrystalshight++;
      if(rhtj<0) nbcrystalssmallt++;
      //Timing test (likelihood ratio), only for seeds with large R (100 cm) and for crystals with et>5, 
      //This targets IT beam halo (t around - 1ns) 
      if(rhtj>5){
      double corrt_j = rhtj + sqrt(rhposj.x()*rhposj.x()+rhposj.y()*rhposj.y() + 320.*320.)/30. - 320./30.;
      timediscriminator+=log( TMath::Gaus(corrt_j,0,0.4,false)/TMath::Gaus(corrt_j,0.3,0.4,false) );
      clustersize++;
      }
      
    }
    //Isolation condition
    if(!isiso) continue;
    
    //Calculate H2/E
    //Only second hcal layer is considered as it can happen that a shower initiated in EE reaches HCAL first layer
    double h2oe(0);
    for(size_t jhit = 0; jhit<hbherechitcoll->size(); ++ jhit){      
      const HBHERecHit & rechitj = (*hbherechitcoll)[ jhit ];
      math::XYZPoint rhposj = getPosition(rechitj.id(),vtx);

      //Ask the hits to be in the same endcap
      if(rhposj.z()*rhpos.z()<0)continue;
      //Selects only second HCAL layer
      if(abs(rhposj.z())<425) continue;
      
      double rhetj = rechitj.energy()/cosh(rhposj.eta());
      if(rhetj<2) continue;
      
      double phij = rhposj.phi();
      if(abs(deltaPhi(phi,phij))>0.4 ) continue;
      
      double rhrj = sqrt(rhposj.x()*rhposj.x()+rhposj.y()*rhposj.y());
      if(abs(rhr-rhrj)>50) continue;

      h2oe+=rhetj/etcluster;
    }
    //H/E condition
    if(h2oe>0.1) continue; 
        

    clustercand.SetClusterEt(etcluster); 
    clustercand.SetSeedEt(rhet); 
    clustercand.SetSeedEta(eta); 
    clustercand.SetSeedPhi(phi); 
    clustercand.SetSeedZ(rhpos.Z());
    clustercand.SetSeedR(sqrt(rhpos.x()*rhpos.x()+rhpos.y()*rhpos.y())); 
    clustercand.SetSeedTime(rechit.time()); 
    clustercand.SetH2overE(h2oe);
    clustercand.SetNbEarlyCrystals(nbcrystalssmallt);
    clustercand.SetNbLateCrystals(nbcrystalshight);
    clustercand.SetClusterSize(clustersize);
    clustercand.SetTimeDiscriminator(timediscriminator);
    clustercand.SetBeamHaloRecHitsCandidates(bhrhcandidates);
    TheHaloClusterCandsEE.push_back(clustercand);
  }

  return  TheHaloClusterCandsEE;
} 




 

std::vector<HaloClusterCandidateHB> CSCHaloAlgo::GetHaloClusterCandidateHB(edm::Handle<EcalRecHitCollection>& ecalrechitcoll, edm::Handle<HBHERecHitCollection>& hbherechitcoll,float et_thresh_seedrh){

  std::vector<HaloClusterCandidateHB> TheHaloClusterCandsHB;

  reco::Vertex::Point vtx(0,0,0);

  for(size_t ihit = 0; ihit<hbherechitcoll->size(); ++ ihit){
    HaloClusterCandidateHB  clustercand;
    
    const HBHERecHit & rechit = (*hbherechitcoll)[ ihit ];
    math::XYZPoint rhpos = getPosition(rechit.id(),vtx);
    //Et condition
    double rhet = rechit.energy()/cosh(rhpos.eta());
    if(rhet<et_thresh_seedrh) continue;
    if(abs(rhpos.z())>380) continue;
    double eta = rhpos.eta();
    double phi = rhpos.phi();
    
    bool isiso = true;
    double etcluster(0);
    int nbtowerssameeta(0);
    double timediscriminatorITBH(0),timediscriminatorOTBH(0);
    double etstrip_phiseedplus1(0), etstrip_phiseedminus1(0);

    //Building the cluster 
    edm::RefVector<HBHERecHitCollection> bhrhcandidates;
    for(size_t jhit = 0; jhit<hbherechitcoll->size(); ++ jhit){
      const HBHERecHit & rechitj = (*hbherechitcoll)[ jhit ];
      HBHERecHitRef rhRef(hbherechitcoll,jhit);
      math::XYZPoint rhposj = getPosition(rechitj.id(),vtx);
      double rhetj = rechitj.energy()/cosh(rhposj.eta());
      if(rhetj<2) continue;
      if(abs(rhposj.z())>380) continue;
      double etaj = rhposj.eta();
      double phij = rhposj.phi();
      if(abs(eta-etaj)>0.4) continue;//This means +/-4 towers in eta 
      if(abs(deltaPhi(phi,phij))>0.2) continue;//This means +/-2 towers in phi 
      if(abs(deltaPhi(phi,phij))>0.1&&abs(eta-etaj)<0.2){isiso=false;break;}//The strip should be isolated
      if(abs(deltaPhi(phi,phij))>0.1)continue;
      if(abs(deltaPhi(phi,phij))<0.05) nbtowerssameeta++;
      if(deltaPhi(phi,phij)>0.05) etstrip_phiseedplus1+=rhetj;
      if(deltaPhi(phi,phij)<-0.05) etstrip_phiseedminus1+=rhetj;
      
      etcluster+=rhetj;
      //Timing discriminator
      //We assign a weight to the rechit defined as: 
      //Log10(Et)*f(T,R,Z)
      //where f(T,R,Z) is the separation curve between halo-like and IP-like times.
      //The time difference between a deposit from a outgoing IT halo and a deposit coming from a particle emitted at the IP is given by:  
      //dt= ( - sqrt(R^2+z^2) + |z| )/c 
      // For OT beam halo, the time difference is: 
      //dt= ( 25 + sqrt(R^2+z^2) + |z| )/c 
      //only consider the central part of HB as things get hard at large z.
      //The best fitted value for R leads to 240 cm (IT) and 330 cm (OT)
      double rhtj = rechitj.time();
      timediscriminatorITBH+= TMath::Log10( rhetj )* ( rhtj +0.5*(sqrt(240.*240.+rhposj.z()*rhposj.z()) -abs(rhposj.z()))/30);
      if(abs(rhposj.z())<300) timediscriminatorOTBH+= TMath::Log10( rhetj )* ( rhtj -0.5*(25-(sqrt(330.*330.+rhposj.z()*rhposj.z()) +abs(rhposj.z()))/30) );
      bhrhcandidates.push_back(rhRef);
    }
    //Isolation conditions
    if(!isiso) continue;
    if(etstrip_phiseedplus1/etcluster>0.2&& etstrip_phiseedminus1/etcluster>0.2) continue;
    
    //Calculate E/H
    double eoh(0);
    for(size_t jhit = 0; jhit<ecalrechitcoll->size(); ++ jhit){
      const EcalRecHit & rechitj = (*ecalrechitcoll)[ jhit ];
      math::XYZPoint rhposj = getPosition(rechitj.id(),vtx);
      double rhetj = rechitj.energy()/cosh(rhposj.eta());
      if(rhetj<2) continue;
      double etaj = rhposj.eta();
      double phij = rhposj.phi();
      if(abs(eta-etaj)>0.2) continue;
      if(abs(deltaPhi(phi,phij))>0.2) continue;
      eoh+=rhetj/etcluster;
    }
    //E/H condition
    if(eoh>0.1) continue; 
        

    clustercand.SetClusterEt(etcluster); 
    clustercand.SetSeedEt(rhet); 
    clustercand.SetSeedEta(eta); 
    clustercand.SetSeedPhi(phi); 
    clustercand.SetSeedZ(rhpos.Z());
    clustercand.SetSeedR(sqrt(rhpos.x()*rhpos.x()+rhpos.y()*rhpos.y())); 
    clustercand.SetSeedTime(rechit.time()); 
    clustercand.SetEoverH(eoh);
    clustercand.SetNbTowersInEta(nbtowerssameeta);
    clustercand.SetEtStripPhiSeedPlus1(etstrip_phiseedplus1);
    clustercand.SetEtStripPhiSeedMinus1(etstrip_phiseedminus1);
    clustercand.SetTimeDiscriminatorITBH(timediscriminatorITBH);
    clustercand.SetTimeDiscriminatorOTBH(timediscriminatorOTBH);
    clustercand.SetBeamHaloRecHitsCandidates(bhrhcandidates);

    TheHaloClusterCandsHB.push_back(clustercand);
  }

  return  TheHaloClusterCandsHB;
} 


std::vector<HaloClusterCandidateHE> CSCHaloAlgo::GetHaloClusterCandidateHE(edm::Handle<EcalRecHitCollection>& ecalrechitcoll, edm::Handle<HBHERecHitCollection>& hbherechitcoll,float et_thresh_seedrh){

  std::vector<HaloClusterCandidateHE> TheHaloClusterCandsHE;

  reco::Vertex::Point vtx(0,0,0);

  for(size_t ihit = 0; ihit<hbherechitcoll->size(); ++ ihit){
    HaloClusterCandidateHE  clustercand;
    
    const HBHERecHit & rechit = (*hbherechitcoll)[ ihit ];
    math::XYZPoint rhpos = getPosition(rechit.id(),vtx);
    //Et condition
    double rhet = rechit.energy()/cosh(rhpos.eta());
    if(rhet<et_thresh_seedrh) continue;
    if(abs(rhpos.z())<380) continue;
    double eta = rhpos.eta();
    double phi = rhpos.phi();
    double rhr = sqrt(rhpos.x()*rhpos.x()+rhpos.y()*rhpos.y());
    bool isiso = true;
    double etcluster(0),hdepth1(0);
    int clustersize(0);
    double etstrip_phiseedplus1(0), etstrip_phiseedminus1(0);

    //Building the cluster 
    edm::RefVector<HBHERecHitCollection> bhrhcandidates;
    for(size_t jhit = 0; jhit<hbherechitcoll->size(); ++ jhit){
      const HBHERecHit & rechitj = (*hbherechitcoll)[ jhit ];
      HBHERecHitRef rhRef(hbherechitcoll,jhit);
      math::XYZPoint rhposj = getPosition(rechitj.id(),vtx);
      double rhetj = rechitj.energy()/cosh(rhposj.eta());
      if(rhetj<2) continue;
      if(abs(rhposj.z())<380) continue;
      if(rhpos.z()*rhposj.z()<0) continue;
      double phij = rhposj.phi();
      if(abs(deltaPhi(phi,phij))>0.4) continue;
      double rhrj = sqrt(rhposj.x()*rhposj.x()+rhposj.y()*rhposj.y());
      if(abs( rhr-rhrj )>50) continue;
      if(abs(deltaPhi(phi,phij))>0.2 ||abs( rhr-rhrj )>20 ){isiso=false;break;}//The deposit should be isolated
      if(deltaPhi(phi,phij)>0.05) etstrip_phiseedplus1+=rhetj;
      if(deltaPhi(phi,phij)<-0.05) etstrip_phiseedminus1+=rhetj;
      clustersize++;
      etcluster+=rhetj;
      if(abs( rhposj.z())<405 )hdepth1+=rhetj;
      //No timing condition for now in HE
      bhrhcandidates.push_back(rhRef);
    }
    //Isolation conditions
    if(!isiso) continue;
    if(etstrip_phiseedplus1/etcluster>0.1&& etstrip_phiseedminus1/etcluster>0.1) continue;
    
    //Calculate E/H
    double eoh(0);
    for(size_t jhit = 0; jhit<ecalrechitcoll->size(); ++ jhit){
      const EcalRecHit & rechitj = (*ecalrechitcoll)[ jhit ];
      math::XYZPoint rhposj = getPosition(rechitj.id(),vtx);
      double rhetj = rechitj.energy()/cosh(rhposj.eta());
      if(rhetj<2) continue;
      if(rhpos.z()*rhposj.z()<0) continue;
      double etaj = rhposj.eta();
      double phij = rhposj.phi();
      double dr = sqrt(abs(eta-etaj)*abs(eta-etaj)+deltaPhi(phi,phij)*deltaPhi(phi,phij));
      if(dr>0.3) continue;

      eoh+=rhetj/etcluster;
    }
    //E/H condition
    if(eoh>0.1) continue; 
        

    clustercand.SetClusterEt(etcluster); 
    clustercand.SetSeedEt(rhet); 
    clustercand.SetSeedEta(eta); 
    clustercand.SetSeedPhi(phi); 
    clustercand.SetSeedZ(rhpos.Z());
    clustercand.SetSeedR(sqrt(rhpos.x()*rhpos.x()+rhpos.y()*rhpos.y())); 
    clustercand.SetSeedTime(rechit.time()); 
    clustercand.SetEoverH(eoh);
    clustercand.SetH1overH123(hdepth1/etcluster);
    clustercand.SetClusterSize(clustersize);
    clustercand.SetEtStripPhiSeedPlus1(etstrip_phiseedplus1);
    clustercand.SetEtStripPhiSeedMinus1(etstrip_phiseedminus1);
    clustercand.SetTimeDiscriminator(0);
    clustercand.SetBeamHaloRecHitsCandidates(bhrhcandidates);

    TheHaloClusterCandsHE.push_back(clustercand);
  }

  return  TheHaloClusterCandsHE;
} 





bool CSCHaloAlgo::SegmentMatchingEB(reco::CSCHaloData & thehalodata, std::vector<HaloClusterCandidateEB>haloclustercands, float iZ, float iR, float iT, float iPhi, bool ishlt){
  bool rhmatchingfound =false;
  for(size_t i =0; i<haloclustercands.size(); i++){
   
    if(!ApplyMatchingCuts("EB",ishlt, haloclustercands[i].GetSeedEt(), iZ, haloclustercands[i].GetSeedZ(),iR, haloclustercands[i].GetSeedR(), iT,haloclustercands[i].GetSeedTime(), iPhi, haloclustercands[i].GetSeedPhi()))continue;
    rhmatchingfound=true;
    edm::RefVector<EcalRecHitCollection> bhrhcandidates = haloclustercands[i].GetBeamHaloRecHitsCandidates();
    AddtoBeamHaloEBEERechits(bhrhcandidates, thehalodata,true);
  }
  return rhmatchingfound;
}


bool CSCHaloAlgo::SegmentMatchingEE(reco::CSCHaloData & thehalodata, std::vector<HaloClusterCandidateEE>haloclustercands, float iZ, float iR, float iT, float iPhi, bool ishlt){
  bool rhmatchingfound =false;
  for(size_t i =0; i<haloclustercands.size(); i++){
   
    if(!ApplyMatchingCuts("EE",ishlt, haloclustercands[i].GetSeedEt(), iZ, haloclustercands[i].GetSeedZ(),iR, haloclustercands[i].GetSeedR(), iT,haloclustercands[i].GetSeedTime(), iPhi, haloclustercands[i].GetSeedPhi()))continue;
    rhmatchingfound=true;
    edm::RefVector<EcalRecHitCollection> bhrhcandidates = haloclustercands[i].GetBeamHaloRecHitsCandidates();
    AddtoBeamHaloEBEERechits(bhrhcandidates, thehalodata,false);
  }
  return rhmatchingfound;
}

bool CSCHaloAlgo::SegmentMatchingHB(reco::CSCHaloData & thehalodata, std::vector<HaloClusterCandidateHB>haloclustercands, float iZ, float iR, float iT, float iPhi, bool ishlt){
  bool rhmatchingfound =false;
  for(size_t i =0; i<haloclustercands.size(); i++){
   
    if(!ApplyMatchingCuts("HB",ishlt, haloclustercands[i].GetSeedEt(), iZ, haloclustercands[i].GetSeedZ(),iR, haloclustercands[i].GetSeedR(), iT,haloclustercands[i].GetSeedTime(), iPhi, haloclustercands[i].GetSeedPhi()))continue;
    rhmatchingfound=true;
    edm::RefVector<HBHERecHitCollection> bhrhcandidates = haloclustercands[i].GetBeamHaloRecHitsCandidates();
    AddtoBeamHaloHBHERechits(bhrhcandidates, thehalodata);
  }
  return rhmatchingfound;
}

bool CSCHaloAlgo::SegmentMatchingHE(reco::CSCHaloData & thehalodata, std::vector<HaloClusterCandidateHE>haloclustercands, float iZ, float iR, float iT, float iPhi, bool ishlt){
  bool rhmatchingfound =false;
  for(size_t i =0; i<haloclustercands.size(); i++){
   
    if(!ApplyMatchingCuts("HE",ishlt, haloclustercands[i].GetSeedEt(),  iZ, haloclustercands[i].GetSeedZ(),iR, haloclustercands[i].GetSeedR(), iT,haloclustercands[i].GetSeedTime(), iPhi, haloclustercands[i].GetSeedPhi()))continue;
    rhmatchingfound=true;
    edm::RefVector<HBHERecHitCollection> bhrhcandidates = haloclustercands[i].GetBeamHaloRecHitsCandidates();
    AddtoBeamHaloHBHERechits(bhrhcandidates, thehalodata);
  }
  return rhmatchingfound;
}


bool CSCHaloAlgo::ApplyMatchingCuts(TString subdet, bool ishlt, double rhet, double segZ, double rhZ, double segR, double rhR, double segT, double rhT, double segPhi, double rhPhi){
  //Absolute time wrt BX
  double tBXrh = rhT+sqrt(rhR*rhR+rhZ*rhZ)/30;
  double tBXseg = segT+sqrt(segR*segR+segZ*segZ)/30;
  //Time at z=0, under beam halo hypothesis    
  double tcorseg = tBXseg - abs(segZ)/30;//Outgoing beam halo 
  double tcorsegincbh = tBXseg + abs(segZ)/30;//Ingoing beam halo
  double truedt[4]={1000,1000,1000,1000};
  //There are four types of segments associated to beam halo, test each hypothesis:
  //IT beam halo, ingoing track
  double twindow_seg = 15;
  if(abs(tcorsegincbh) <twindow_seg) truedt[0] =   tBXrh -tBXseg  -abs(rhZ-segZ)/30; 
  //IT beam halo, outgoing track
  if(abs(tcorseg) < twindow_seg) truedt[1] =  tBXseg -tBXrh -abs(rhZ-segZ)/30;  
  //OT beam halo (from next BX), ingoing track
  if(tcorsegincbh> 25-twindow_seg&& abs(tcorsegincbh) <25+twindow_seg) truedt[2] =   tBXrh -tBXseg  -abs(rhZ-segZ)/30; 
  //OT beam halo (from next BX), outgoing track
  if(tcorseg >25-twindow_seg && tcorseg<25+twindow_seg) truedt[3] =   tBXseg -tBXrh -abs(rhZ-segZ)/30; 
  
  
  if(subdet=="EB"){
    if(rhet< et_thresh_rh_eb) return false;
    if(rhet< 20&&ishlt) return false;
    if(abs(deltaPhi(rhPhi,segPhi))>dphi_thresh_segvsrh_eb) return false;
    if(rhR-segR< dr_lowthresh_segvsrh_eb)return false;
    if(rhR-segR> dr_highthresh_segvsrh_eb) return false;
    if(abs(truedt[0])>dt_segvsrh_eb &&abs(truedt[1])>dt_segvsrh_eb &&abs(truedt[2])>dt_segvsrh_eb &&abs(truedt[3])>dt_segvsrh_eb   )return false;
    return true;
  }


  if(subdet=="EE"){
    
    if(rhet< et_thresh_rh_ee) return false;
    if(rhet< 20&&ishlt) return false;
    if(abs(deltaPhi(rhPhi,segPhi))>dphi_thresh_segvsrh_ee) return false;
    if(rhR-segR< dr_lowthresh_segvsrh_ee)return false;
    if(rhR-segR> dr_highthresh_segvsrh_ee) return false;
    if(abs(truedt[0])>dt_segvsrh_ee &&abs(truedt[1])>dt_segvsrh_ee &&abs(truedt[2])>dt_segvsrh_ee &&abs(truedt[3])>dt_segvsrh_ee   )return false;
    return true;
  }

  if(subdet=="HB"){
    
    if(rhet< et_thresh_rh_hb) return false;
    if(abs(deltaPhi(rhPhi,segPhi))>dphi_thresh_segvsrh_hb) return false;
    if(rhR-segR< dr_lowthresh_segvsrh_hb)return false;
    if(rhR-segR> dr_highthresh_segvsrh_hb) return false;
    if(abs(truedt[0])>dt_segvsrh_hb &&abs(truedt[1])>dt_segvsrh_hb &&abs(truedt[2])>dt_segvsrh_hb &&abs(truedt[3])>dt_segvsrh_hb   )return false;
    return true;
  }

  if(subdet=="HE"){
    
    if(rhet< et_thresh_rh_he) return false;
    if(abs(deltaPhi(rhPhi,segPhi))>dphi_thresh_segvsrh_he) return false;
    if(rhR-segR< dr_lowthresh_segvsrh_he)return false;
    if(rhR-segR> dr_highthresh_segvsrh_he) return false;
    if(abs(truedt[0])>dt_segvsrh_he &&abs(truedt[1])>dt_segvsrh_he &&abs(truedt[2])>dt_segvsrh_he &&abs(truedt[3])>dt_segvsrh_he   )return false;
    return true;
  }


  return false;
}
