#include "RecoMET/METAnalyzers/interface/CSCHaloFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace std;
using namespace edm;
using namespace reco;
CSCHaloFilter::CSCHaloFilter(const edm::ParameterSet & iConfig)
{
  IT_L1MuGMTReadout = iConfig.getParameter<edm::InputTag>("L1MuGMTReadoutLabel");
  IT_ALCTDigi = iConfig.getParameter<edm::InputTag>("ALCTDigiLabel");
  IT_CollisionMuon = iConfig.getParameter<edm::InputTag>("CollisionMuonLabel");
  IT_SACosmicMuon = iConfig.getParameter<edm::InputTag>("SACosmicMuonLabel");
  IT_CSCRecHit = iConfig.getParameter<edm::InputTag>("CSCRecHitLabel");
  IT_CSCSegment = iConfig.getParameter<edm::InputTag>("CSCSegmentLabel");
  IT_CSCHaloData = iConfig.getParameter<edm::InputTag>("CSCHaloDataLabel");
  IT_BeamHaloSummary = iConfig.getParameter<edm::InputTag>("BeamHaloSummaryLabel");


  deta_threshold = (float) iConfig.getParameter<double>("Deta");
  dphi_threshold = (float) iConfig.getParameter<double>("Dphi");
  min_inner_radius = (float) iConfig.getParameter<double>("InnerRMin");
  max_inner_radius = (float) iConfig.getParameter<double>("InnerRMax");
  min_outer_radius = (float) iConfig.getParameter<double>("OuterRMin");
  max_outer_radius = (float) iConfig.getParameter<double>("OuterRMax");
  norm_chi2_threshold = (float) iConfig.getParameter<double>("NormChi2");
  min_outer_theta = (float)iConfig.getParameter<double>("MinOuterMomentumTheta");
  max_outer_theta = (float)iConfig.getParameter<double>("MaxOuterMomentumTheta");
  max_dr_over_dz = (float)iConfig.getParameter<double>("MaxDROverDz");
 
  expected_BX  = (short int) iConfig.getParameter<int>("ExpectedBX") ; 
  
  matching_dphi_threshold =  (float)iConfig.getParameter<double>("MatchingDPhiThreshold");
  matching_deta_threshold =  (float)iConfig.getParameter<double>("MatchingDEtaThreshold");
  matching_dwire_threshold  = iConfig.getParameter<int>("MatchingDWireThreshold");

  FilterCSCLoose = iConfig.getParameter<bool>("FilterCSCLoose");
  FilterCSCTight = iConfig.getParameter<bool>("FilterCSCTight");

  FilterTriggerLevel = iConfig.getParameter<bool>("FilterTriggerLevel");
  FilterDigiLevel = iConfig.getParameter<bool>("FilterDigiLevel");
  FilterRecoLevel = iConfig.getParameter<bool>("FilterRecoLevel");

  min_nHaloTriggers = FilterTriggerLevel ? iConfig.getUntrackedParameter<int>("MinNumberOfHaloTriggers",1) : 99999;
  min_nHaloDigis    = FilterDigiLevel ?  iConfig.getUntrackedParameter<int>("MinNumberOfOutOfTimeDigis",1) : 99999;
  min_nHaloTracks   = FilterRecoLevel ?  iConfig.getUntrackedParameter<int>("MinNumberOfHaloTracks",1) : 99999;

  // Load TrackDetectorAssociator parameters                                                                                                                   
  edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  parameters_.loadParameters( parameters );
}


CSCHaloFilter::~CSCHaloFilter(){}

bool CSCHaloFilter::filter(edm::Event & iEvent, const edm::EventSetup & iSetup) 
{

  if( FilterCSCLoose || FilterCSCTight ) 
    {
      edm::Handle<BeamHaloSummary> TheBeamHaloSummary;
      iEvent.getByLabel(IT_BeamHaloSummary,TheBeamHaloSummary);

      const BeamHaloSummary TheSummary = (*TheBeamHaloSummary.product() );
      
      if( FilterCSCLoose ) 
	return !TheSummary.CSCLooseHaloId();
      else 
	return !TheSummary.CSCTightHaloId();
    }
  

  //Get B-Field
  edm::ESHandle<MagneticField> TheMagneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(TheMagneticField);

  //Get Propagator
  edm::ESHandle<Propagator> propagator;
  iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAny", propagator);
  trackAssociator_.setPropagator(propagator.product());

  //Get CSC Geometry
  edm::ESHandle<CSCGeometry> TheCSCGeometry;
  iSetup.get<MuonGeometryRecord>().get(TheCSCGeometry);

  // Get Collision Muon Collection
  edm::Handle<reco::MuonCollection> TheCollisionMuons;
  iEvent.getByLabel(IT_CollisionMuon,TheCollisionMuons);
    
  //Get Cosmic  Stand-Alone Muons
  edm::Handle<reco::TrackCollection> TheSACosmicMuons;
  iEvent.getByLabel( IT_SACosmicMuon, TheSACosmicMuons);

  //Get CSC Segments
  //edm::Handle<CSCSegmentCollection> TheCSCSegments;
  //iEvent.getByLabel(IT_CSCSegment, TheCSCSegments);
  
  //Get CSC RecHits
  //Handle<CSCRecHit2DCollection> TheCSCRecHits;
  //iEvent.getByLabel(IT_CSCRecHit, TheCSCRecHits);
  
  edm::Handle<reco::CSCHaloData> TheCSCDataHandle;
  iEvent.getByLabel(IT_CSCHaloData,TheCSCDataHandle);


  int nHaloCands  = 0;
  int nHaloDigis  = 0;
  int nHaloTracks = 0;

  if(TheCSCDataHandle.isValid())                                                                                                                        
    {                                                                                                                                                     
      const reco::CSCHaloData CSCData = (*TheCSCDataHandle.product());                                                                                    
      nHaloDigis = CSCData.NumberOfOutOfTimeTriggers() ;                                                                                                    
      nHaloCands = CSCData.NumberOfHaloTriggers();
    }      


  /*
  if( FilterTriggerLevel )
    {
      //Get L1MuGMT 
      edm::Handle < L1MuGMTReadoutCollection > TheL1GMTReadout ;
      iEvent.getByLabel (IT_L1MuGMTReadout, TheL1GMTReadout);
      if( TheL1GMTReadout.isValid() )
	{
	  L1MuGMTReadoutCollection const *MuGMT = TheL1GMTReadout.product ();
	  std::vector < L1MuGMTReadoutRecord > TheRecords = MuGMT->getRecords ();
	  std::vector < L1MuGMTReadoutRecord >::const_iterator iRecord;
	  for (iRecord = TheRecords.begin (); iRecord != TheRecords.end (); iRecord++)
	    {
	      std::vector < L1MuRegionalCand >::const_iterator iCand;
	      std::vector < L1MuRegionalCand > TheCands = iRecord->getCSCCands ();
	      for (iCand = TheCands.begin (); iCand != TheCands.end (); iCand++)
		{
		  if (!(*iCand).empty ())
		    {
		      if ((*iCand).isFineHalo ())
			{
			  float halophi = iCand->phiValue();
			  float haloeta = iCand->etaValue();
			  halophi = halophi > TMath::Pi() ? halophi - 2.*TMath::Pi() : halophi;
			  bool CandIsHalo = true;
			  // Check if halo trigger is faked by any collision muons

			  if( TheCollisionMuons.isValid() )
			    {
			      float dphi = 9999.;
			      float deta = 9999.;
			      for( reco::MuonCollection::const_iterator iMuon = TheCollisionMuons->begin(); iMuon != TheCollisionMuons->end() && CandIsHalo ; iMuon++ )
				{
				  std::vector<TrackDetMatchInfo> info;
				  if( iMuon->isTrackerMuon() && !iMuon->innerTrack().isNull() )
				    info.push_back( trackAssociator_.associate(iEvent, iSetup, *iMuon->innerTrack(), parameters_) );
				  if( iMuon->isStandAloneMuon() && !iMuon->outerTrack().isNull() )
				    {
				      //make sure that this SA muon is not actually a halo-like muon
                                      float theta =  iMuon->outerTrack()->outerMomentum().theta();
                                      float deta = TMath::Abs(iMuon->outerTrack()->outerPosition().eta() - iMuon->outerTrack()->innerPosition().eta());
                                      
				      if( !( theta < min_outer_theta || theta > max_outer_theta) )  //halo-like                        
					if ( deta <= deta_threshold ) //halo-like               	
					  {
					    if( iMuon->isGlobalMuon() || iMuon->isTrackerMuon() ) // NOT SA-Only 
					      info.push_back( trackAssociator_.associate(iEvent, iSetup, *iMuon->outerTrack(), parameters_) );
					  }
				    }
				  if ( iMuon->isGlobalMuon() && !iMuon->globalTrack().isNull() )
				    info.push_back(trackAssociator_.associate(iEvent, iSetup, *iMuon->globalTrack(), parameters_) );
				  
				  for(unsigned int i = 0 ; i < info.size(); i++ )
				    {
				      for( std::vector<TAMuonChamberMatch>::const_iterator chamber=info[i].chambers.begin();
					   chamber!=info[i].chambers.end(); chamber++ ){
					if( chamber->detector() != MuonSubdetId::CSC ) continue;
					  
					  for( std::vector<TAMuonSegmentMatch>::const_iterator segment = chamber->segments.begin();
					       segment != chamber->segments.end(); segment++ ) {

					    float eta_ = segment->segmentGlobalPosition.eta();	
					    float phi_ = segment->segmentGlobalPosition.phi();	
					    float test_dphi = TMath::Abs(phi_ - halophi);
					    test_dphi = TMath::ACos( TMath::Cos( test_dphi ) );
					    float test_deta = TMath::Abs(eta_ - haloeta);
					    dphi = dphi < test_dphi ? dphi : test_dphi;
					    deta = deta < test_deta ? deta : test_deta;
					  }
				      }
				    }
				  if ( dphi < matching_dphi_threshold && deta < matching_deta_threshold ) //collision likely caused it
				    CandIsHalo = false; 
				}
			    }
			  if(CandIsHalo)
			    nHaloCands++;
			}
		    }
		}
	    }
	}
      else
	{
	  LogWarning("Collection Not Found") << "You have requested Trigger-level filtering, but the L1MuGMTReadoutCollection does not appear"
					     << "to be in the event! Trigger-level filtering will be disabled" ;

	  FilterTriggerLevel = false;  //NO TRIGGER DECISION CAN BE MADE
	}
    }

  if(FilterDigiLevel)
    {
      //Get Chamber Anode Trigger Information                                                                                                                      
      edm::Handle<CSCALCTDigiCollection> TheALCTs;
   iEvent.getByLabel (IT_ALCTDigi, TheALCTs);
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
		      bool DigiIsHalo = true;
		      // MATCHING CHECK NOT AVAILABLE UNTIL 36X 
			 
		      int digi_endcap  = detId.endcap();
		      int digi_station = detId.station();
		      int digi_ring    = detId.ring();
		      int digi_chamber = detId.chamber();
		      int digi_wire    = digiIt->getKeyWG();
		      
		      if( digi_station == 1 && digi_ring == 4 )   //hack
		      digi_ring = 1;
		      
		      int dwire = 999.;
		      if( TheCollisionMuons.isValid() ) 
		      {
		      for( reco::MuonCollection::const_iterator iMuon = TheCollisionMuons->begin(); iMuon != TheCollisionMuons->end() && DigiIsHalo ; iMuon++ )
		      {
		      
		      std::vector<TrackDetMatchInfo> info;
		      if( iMuon->isTrackerMuon() && !iMuon->innerTrack().isNull() )
		      info.push_back( trackAssociator_.associate(iEvent, iSetup, *iMuon->innerTrack(), parameters_) );
		      if( iMuon->isStandAloneMuon() && !iMuon->outerTrack().isNull() )
		      {
		      //make sure that this SA muon is not actually a halo-like muon                           
		      float theta =  iMuon->outerTrack()->outerMomentum().theta();
		      float deta = TMath::Abs(iMuon->outerTrack()->outerPosition().eta() - iMuon->outerTrack()->innerPosition().eta());
		      
		      if( !( theta < min_outer_theta || theta > max_outer_theta ) )
		      if ( deta > deta_threshold ) //halo-like 
		      info.push_back( trackAssociator_.associate(iEvent, iSetup, *iMuon->outerTrack(), parameters_) );
		      }
		      if (iMuon->isGlobalMuon() && !iMuon->globalTrack().isNull() )
		      info.push_back(trackAssociator_.associate(iEvent, iSetup, *iMuon->globalTrack(), parameters_) );
		      for(unsigned int i = 0 ; i < info.size(); i++ )
		      {
		      for( std::vector<TAMuonChamberMatch>::const_iterator chamber=info[i].chambers.begin();
		      chamber!=info[i].chambers.end(); chamber++ ){
		      if( chamber->detector() != MuonSubdetId::CSC ) continue;
		      
		      for( std::vector<TAMuonSegmentMatch>::const_iterator segment = chamber->segments.begin();
		      segment != chamber->segments.end(); segment++ ) {
		      
		      float eta_ = segment->segmentGlobalPosition.eta();	
		      float phi_ = segment->segmentGlobalPosition.phi();	
		      }
		      }
		      }
		      if( dwire <=  matching_dwire_threshold ) 
		      DigiIsHalo = false;
		      }

		      if( DigiIsHalo )
			nHaloDigis++;
		    }
		}
	    }
	}

	else if( TheCSCDataHandle.isValid() )
	{
	//FOR >= 36X 
	
	If the user wants to use the info from the ALCT digis 
	but they aren't in the event, then the reco::CSCHaloData
	object can be used, albeit in limited scope,i.e.,
	ExpectedBX == 3 and no matching to collision muon hits
	is done until CMSSW_3_7_0.  
	
	//NOTE : THIS BLOCK OF CODE ONLY WORKS FOR >= 3_6_0 
	if(TheCSCDataHandle.isValid())
	{
	const reco::CSCHaloData CSCData = (*TheCSCDataHandle.product());
	nHaloDigis = CSCData.NumberOfOutOfTimeTriggers() ;
	}
	}

      else
	{
	  LogWarning("Collection Not Found") << "You have requested Digi-level filtering, but the CSCALCTDigiCollection does not appear"
					     << "to be in the event! Digl-level filtering will be disabled" ;   
	  
	  FilterDigiLevel = false ; // NO DIGI LEVEL DECISION CAN BE MADE
	}
    }

*/
  
  if(FilterRecoLevel)
    {
      if(TheSACosmicMuons.isValid())
	{
	  for( reco::TrackCollection::const_iterator iTrack = TheSACosmicMuons->begin() ; iTrack != TheSACosmicMuons->end() ; iTrack++ )
	    {
	      bool TrackIsHalo = true;;
	      // Calculate global phi coordinate for central most rechit in the track
	      float innermost_global_z = 1500.;
	      float outermost_global_z = 0.;
	      
	      GlobalPoint InnerMostGlobalPosition;  // smallest abs(z)
	      GlobalPoint OuterMostGlobalPosition;  // largest abs(z)
	      
	      int nCSCHits = 0;
	      for(unsigned int j = 0 ; j < iTrack->extra()->recHits().size(); j++ )
		{
		  edm::Ref<TrackingRecHitCollection> hit( iTrack->extra()->recHits(), j );
		  if( !hit->isValid() ) continue;
		  DetId TheDetUnitId(hit->geographicalId());
		  if( TheDetUnitId.det() != DetId::Muon ) continue;

		  if( TheDetUnitId.subdetId() != MuonSubdetId::CSC ) continue;

		  const GeomDetUnit *TheUnit = TheCSCGeometry->idToDetUnit(TheDetUnitId);
		  LocalPoint TheLocalPosition = hit->localPosition();  
		  const BoundPlane TheSurface = TheUnit->surface();
		  const GlobalPoint TheGlobalPosition = TheSurface.toGlobal(TheLocalPosition);

		  float z = TheGlobalPosition.z();
		  // Get consituent rechit closest to calorimetry
		  if( TMath::Abs(z) < innermost_global_z )
		    {
		      innermost_global_z = TMath::Abs(z);
		      InnerMostGlobalPosition = GlobalPoint( TheGlobalPosition);
		    }
		  // Get constituent rechit farthest from calorimetry
		  if( TMath::Abs(z) > outermost_global_z )
		    {
		      outermost_global_z = TMath::Abs(z);
		      OuterMostGlobalPosition = GlobalPoint( TheGlobalPosition );
		    }
		  nCSCHits ++;
		}

	      if( nCSCHits < 3 ) continue; // This needs to be optimized 
	      
	      float deta = TMath::Abs( OuterMostGlobalPosition.eta() - InnerMostGlobalPosition.eta() );
	      float dphi = TMath::ACos( TMath::Cos( OuterMostGlobalPosition.phi() - InnerMostGlobalPosition.phi() ) ) ;
	      float theta = iTrack->outerMomentum().theta();
	      float innermost_x = InnerMostGlobalPosition.x() ;
	      float innermost_y = InnerMostGlobalPosition.y();
	      float outermost_x = OuterMostGlobalPosition.x();
	      float outermost_y = OuterMostGlobalPosition.y();
	      float innermost_r = TMath::Sqrt(innermost_x *innermost_x + innermost_y * innermost_y );
	      float outermost_r = TMath::Sqrt(outermost_x *outermost_x + outermost_y * outermost_y );
	      
	      float dr = TMath::Abs(innermost_r - outermost_r) ;
	      float dz = TMath::Abs(InnerMostGlobalPosition.z()  - OuterMostGlobalPosition.z() );
	      //float detadz = deta / ( innermost_global_z - outermost_global_z ) ;
	      
	      if( deta < deta_threshold )
		TrackIsHalo = false;
	      else if( theta > min_outer_theta && theta < max_outer_theta )
		TrackIsHalo = false;
	      else if( dphi > dphi_threshold )
		TrackIsHalo = false;
	      else if( innermost_r < min_inner_radius )
		TrackIsHalo = false;
	      else if( innermost_r > max_inner_radius )
		TrackIsHalo = false;
	      else if( outermost_r < min_outer_radius )
		TrackIsHalo = false;
	      else if( outermost_r > max_outer_radius )
		TrackIsHalo = false;
	      else if( iTrack->normalizedChi2() > norm_chi2_threshold )
		TrackIsHalo = false;
	      else if( dz )
		{
		  if( dr/dz > max_dr_over_dz ) 
		    {
		      TrackIsHalo = false;
		    }
		}
	      
	      if( TrackIsHalo )
		{
		  nHaloTracks++;
		  /*
		    cout << "deta " << deta << endl;
		    cout << "dphi " << dphi << endl;
		    cout << "theta " << theta << endl;
		    cout << "innermost_r " << innermost_r << endl;
		    cout << "outermost_r " << outermost_r << endl;
		    cout << "norm_chi2 " << iTrack->normalizedChi2() << endl;
		    cout << "NValidHits " << iTrack->numberOfValidHits() << endl;
		    cout << "nCSCHits " << nCSCHits << endl;
		    cout << "deta/dz " << detadz << endl;
		    cout << "dr/dz " << dr/dz << endl;
		  */
		}
	    }
	}
      else
	{
	  LogWarning("Collection Not Found") << "You have requested Reco-level filtering, but the cosmic stand-alone muon collection does not appear"
					     << "to be in the event! Reco-level filtering will be disabled" ;   

	  FilterRecoLevel = false; //NO RECO LEVEL DECISION CAN BE MADE
	}
    }

  if(FilterRecoLevel && FilterDigiLevel && FilterTriggerLevel)
    return !( nHaloTracks >= min_nHaloTracks && nHaloDigis >= min_nHaloDigis && nHaloCands >= min_nHaloTriggers );
  else if( FilterRecoLevel && FilterDigiLevel )
    return !(nHaloTracks>= min_nHaloTracks && nHaloDigis >= min_nHaloDigis);
  else if( FilterRecoLevel && FilterTriggerLevel ) 
    return !(nHaloTracks>= min_nHaloTracks && nHaloCands >= min_nHaloTriggers );
  else if( FilterDigiLevel && FilterTriggerLevel )
    return !( nHaloDigis >= min_nHaloDigis && nHaloCands >= min_nHaloTriggers );
  else if( FilterDigiLevel ) 
    return  !(nHaloDigis >= min_nHaloDigis);
  else if( FilterRecoLevel ) 
    return !( nHaloTracks >= min_nHaloTracks );
  else if( FilterTriggerLevel ) 
    return !( nHaloCands >= min_nHaloTriggers) ;
  else
    return true;
  


}
  

DEFINE_FWK_MODULE(CSCHaloFilter);
