#include <memory>
#include <string>
#include <iostream>
#include <fstream>

#include "AnalysisExamples/SiStripDetectorPerformance/interface/TIFNtupleMaker.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/LTCDigi/interface/LTCDigi.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "MagneticField/Engine/interface/MagneticField.h"

using namespace std;

//Constructor

TIFNtupleMaker::TIFNtupleMaker(edm::ParameterSet const& conf) : 
  conf_(conf), 
  filename_(conf.getParameter<std::string>("fileName")),
  oSiStripDigisLabel_( conf.getUntrackedParameter<std::string>( "oSiStripDigisLabel")),
  oSiStripDigisProdInstName_( conf.getUntrackedParameter<std::string>( "oSiStripDigisProdInstName")),
  // Remove candidate
  // ----------------
  //  bUseLTCDigis_( conf.getUntrackedParameter<bool>( "bUseLTCDigis")),
  // ----------------
  dCROSS_TALK_ERR( conf.getUntrackedParameter<double>( "dCrossTalkErr")),
  bTriggerDT( false),
  bTriggerCSC( false),
  bTriggerRBC1( false),
  bTriggerRBC2( false),
  bTriggerRPC( false)
  //m_oSiStripNoiseService( conf_)
{
  anglefinder_=new  TrackLocalAngleTIF(conf);  
}

//BeginJob

void TIFNtupleMaker::beginJob(const edm::EventSetup& c){

  hFile = new TFile (filename_.c_str(), "RECREATE" );
  
  // Investigate this two
  // --------------------
  hphi = new TH1F("hphi","Phi distribution",20,-3.14,3.14);
  hnhit = new TH1F("hnhit","Number of Hits per Track ",18,2,20);
  // --------------------
  hEventTrackNumber = new TH1F("hEventTrackNumber","Number of tracks in the event ",10,0,10);

  hwvst = new TProfile("WidthvsTrackProjection","Cluster width vs track projection ",120,-60.,60.);
  fitfunc = new TF1("fitfunc","[1]*((x-[0])^2)+[2]",-30,30); 
   
  TIFNtupleMakerTree = new TTree("TIFNtupleMakerTree","SiStrip LorentzAngle tree");
  TIFNtupleMakerTree->Branch("run", &run, "run/I");
  TIFNtupleMakerTree->Branch( "eventcounter", &eventcounter, "eventcounter/I");
  TIFNtupleMakerTree->Branch("event", &event, "event/I");
  TIFNtupleMakerTree->Branch("module", &module, "module/I");
  TIFNtupleMakerTree->Branch("type", &type, "type/I");
  TIFNtupleMakerTree->Branch("layer", &layer, "layer/I");
  TIFNtupleMakerTree->Branch("string", &string, "string/I");
  TIFNtupleMakerTree->Branch("rod", &rod, "rod/I");
  TIFNtupleMakerTree->Branch("extint", &extint, "extint/I");
  TIFNtupleMakerTree->Branch("size", &size, "size/I");
  TIFNtupleMakerTree->Branch("angle", &angle, "angle/F");
  // Adding phi and theta
  TIFNtupleMakerTree->Branch("tk_phi", &tk_phi, "tk_phi/F");
  TIFNtupleMakerTree->Branch("tk_theta", &tk_theta, "tk_theta/F");
  // --------------------
  TIFNtupleMakerTree->Branch("sign", &sign, "sign/I");
  TIFNtupleMakerTree->Branch("bwfw", &bwfw, "bwfw/I");
  TIFNtupleMakerTree->Branch("wheel", &wheel, "wheel/I");
  TIFNtupleMakerTree->Branch("monostereo", &monostereo, "monostereo/I");
  TIFNtupleMakerTree->Branch("stereocorrection", &stereocorrection, "stereocorrection/F");
  TIFNtupleMakerTree->Branch("localmagfield", &localmagfield, "localmagfield/F");
  TIFNtupleMakerTree->Branch("momentum", &momentum, "momentum/F");
  TIFNtupleMakerTree->Branch("pt", &pt, "pt/F");
  TIFNtupleMakerTree->Branch("charge", &charge, "charge/I");
  TIFNtupleMakerTree->Branch("eta", &eta, "eta/F");
  TIFNtupleMakerTree->Branch("phi", &phi, "phi/F");
  TIFNtupleMakerTree->Branch("hitspertrack", &hitspertrack, "hitspertrack/I");
  TIFNtupleMakerTree->Branch( "normchi2", &normchi2, "normchi2/F");
  TIFNtupleMakerTree->Branch( "chi2", &chi2, "chi2/F");
  TIFNtupleMakerTree->Branch( "ndof", &ndof, "ndof/F");
  TIFNtupleMakerTree->Branch( "bTrack", &bTrack, "bTrack/O");
  TIFNtupleMakerTree->Branch( "numberoftracks", &numberoftracks, "numberoftracks/I");
  TIFNtupleMakerTree->Branch( "clusterpos", &clusterpos, "clusterpos/F");
  TIFNtupleMakerTree->Branch( "clustereta", &clustereta, "clustereta/F");
  TIFNtupleMakerTree->Branch( "clusterchg", &clusterchg, "clusterchg/F");
  TIFNtupleMakerTree->Branch( "clusterchgl", &clusterchgl, "clusterchgl/F");
  TIFNtupleMakerTree->Branch( "clusterchgr", &clusterchgr, "clusterchgr/F");
  TIFNtupleMakerTree->Branch( "clusternoise", &clusternoise, "clusternoise/F");
  TIFNtupleMakerTree->Branch( "clusterbarycenter", &clusterbarycenter, "clusterbarycenter/F");
  TIFNtupleMakerTree->Branch( "clustermaxchg", &clustermaxchg, "clustermaxchg/F");
  TIFNtupleMakerTree->Branch( "clusterseednoise", &clusterseednoise, "clusterseednoise/F");
  TIFNtupleMakerTree->Branch( "clustercrosstalk", &clustercrosstalk, "clustercrosstalk/F");
  // Remove candidate
  // ----------------
  // Trigger Bits
  //   TIFNtupleMakerTree->Branch( "bTriggerDT",   &bTriggerDT,	  "bTriggerDT/O");
  //   TIFNtupleMakerTree->Branch( "bTriggerCSC",  &bTriggerCSC,  "bTriggerCSC/O");
  //   TIFNtupleMakerTree->Branch( "bTriggerRBC1", &bTriggerRBC1, "bTriggerRBC1/O");
  //   TIFNtupleMakerTree->Branch( "bTriggerRBC2", &bTriggerRBC2, "bTriggerRBC2/O");
  //   TIFNtupleMakerTree->Branch( "bTriggerRPC",  &bTriggerRPC,  "bTriggerRPC/O");
  //   TIFNtupleMakerTree->Branch( "dLclX", &dLclX, "dLclX/F");
  //   TIFNtupleMakerTree->Branch( "dLclY", &dLclY, "dLclY/F");
  //   TIFNtupleMakerTree->Branch( "dLclZ", &dLclZ, "dLclZ/F");
  //   TIFNtupleMakerTree->Branch( "dGlbX", &dGlbX, "dGlbX/F");
  //   TIFNtupleMakerTree->Branch( "dGlbY", &dGlbY, "dGlbY/F");
  //   TIFNtupleMakerTree->Branch( "dGlbZ", &dGlbZ, "dGlbZ/F");
  // ----------------
  poTrackTree = new TTree( "TrackTree", "This is a Track specific variables tree");
  poTrackTree->Branch( "run",	         &run,	          "run/I");
  poTrackTree->Branch( "pt",	         &pt,	          "pt/F");
  poTrackTree->Branch( "eta",	         &eta,	          "eta/F");
  poTrackTree->Branch( "phi",	         &phi,	          "phi/F");
  poTrackTree->Branch( "ndof",	         &ndof,	          "ndof/F");
  poTrackTree->Branch( "chi2",	         &chi2,	          "chi2/F");
  poTrackTree->Branch( "event",	         &event,	  "event/I");
  poTrackTree->Branch( "charge",         &charge,	  "charge/I");
  poTrackTree->Branch( "momentum",       &momentum,       "momentum/F");
  poTrackTree->Branch( "eventcounter",   &eventcounter,   "eventcounter/I");
  poTrackTree->Branch( "hitspertrack",   &hitspertrack,   "hitspertrack/I");
  poTrackTree->Branch( "numberoftracks", &numberoftracks, "numberoftracks/I");
  poTrackTree->Branch( "numberofclusters", &numberofclusters, "numberofclusters/I");
  poTrackTree->Branch( "numberoftkclusters", &numberoftkclusters, "numberoftkclusters/I");
  poTrackTree->Branch( "numberofnontkclusters", &numberofnontkclusters, "numberofnontkclusters/I");
  // Remove candidate
  // ----------------
  // Trigger Bits
  //   poTrackTree->Branch( "bTriggerDT",   &bTriggerDT,   "bTriggerDT/O");
  //   poTrackTree->Branch( "bTriggerCSC",  &bTriggerCSC,  "bTriggerCSC/O");
  //   poTrackTree->Branch( "bTriggerRBC1", &bTriggerRBC1, "bTriggerRBC1/O");
  //   poTrackTree->Branch( "bTriggerRBC2", &bTriggerRBC2, "bTriggerRBC2/O");
  //   poTrackTree->Branch( "bTriggerRPC",  &bTriggerRPC,  "bTriggerRPC/O");
  // ----------------

  // New tree for number of tracks
  poTrackNum = new TTree( "TrackNum", "This is Global tree, contains track number");
  poTrackNum->Branch( "numberoftracks", &numberoftracks, "numberoftracks/I");

  eventcounter = 0;
  trackcounter = 0;
  hitcounter   = 0;
  
  edm::ESHandle<MagneticField> esmagfield;
  c.get<IdealMagneticFieldRecord>().get(esmagfield);
  magfield=&(*esmagfield);
    
  edm::ESHandle<TrackerGeometry> estracker;
  c.get<TrackerDigiGeometryRecord>().get(estracker);
  tracker=&(*estracker); 

  // Declare summary histograms
  _summaryHistos();
  // Create directory hierarchy
  _directoryHierarchy();
}

// Virtual destructor needed.

TIFNtupleMaker::~TIFNtupleMaker() {  
  // delete poTrackTree;
}  

// Analyzer: Functions that gets called by framework every event

void TIFNtupleMaker::analyze(const edm::Event& e, const edm::EventSetup& es) {
  //m_oSiStripNoiseService.setESObjects( es);

  run       = e.id().run();
  event     = e.id().event();
  
  eventcounter+=1;
  std::cout << "Event number " << eventcounter << std::endl;

  using namespace edm;
  
  // Step A: Get Inputs 
  
  // Initialize the angle finder
  anglefinder_->init(e,es);
  
  trackhitmap trackhits;
  trackhitmap trackhitsXZ;
  trackhitmap trackhitsYZ;

  trklcldirmap oLclDirs;
  trkglbdirmap oGlbDirs;

    
  //LogDebug("TIFNtupleMaker::analyze")<<"TIF - Getting tracks";
  
  edm::Handle<reco::TrackCollection> trackCollection;
  e.getByLabel( conf_.getParameter<std::string>( "TracksLabel"), trackCollection);
  //e.getByType(trackCollection);

  // Remove candidate
  // ----------------
  //   if( bUseLTCDigis_) {
  //     // Extract Trigger Bits
  //     edm::Handle<LTCDigiCollection> oLTCDigis;
  //     e.getByType( oLTCDigis);
  
  //     // Now loop over all 6 triggers and save their values in corresponding
  //     // boolean variables :)
  //     if( 1 > oLTCDigis->size()) {
  //       LogDebug( "TIFNtupleMaker::analyze")
  // 	<< "[warning] More than one LTCDigis object stored in LTCDigiCollection";
  //     }
  
  //     for( LTCDigiCollection::const_iterator oITER = oLTCDigis->begin();
  // 	 oITER != oLTCDigis->end();
  // 	 ++oITER) {
  
  //       bTriggerDT   |= oITER->HasTriggered( 0);
  //       bTriggerCSC  |= oITER->HasTriggered( 1);
  //       bTriggerRBC1 |= oITER->HasTriggered( 2);
  //       bTriggerRBC2 |= oITER->HasTriggered( 3);
  //       bTriggerRPC  |= oITER->HasTriggered( 4);
  //     }
  //   }
  // --------------------



  // ------------------------------------------------
  // THIS MUST BE CHANGED USING THE TRACK_INFO CLASS,
  // INSTEAD OF LOOPING ON SEEDS
  // ------------------------------------------------
  const reco::TrackCollection *tracks=trackCollection.product();
 
  //LogDebug("TIFNtupleMaker::analyze")<<"TIF - Getting seed";
    
  edm::Handle<TrajectorySeedCollection> seedcoll;
  e.getByLabel( conf_.getParameter<std::string>( "SeedsLabel"), seedcoll);
  //e.getByType(seedcoll);
    
  //LogDebug("TIFNtupleMaker::analyze")<<"TIF - Getting used rechit";

  if((*seedcoll).size()>0){
    if (tracks->size()>0){
       	            
      trackcounter+=tracks->size();
      
      reco::TrackCollection::const_iterator ibeg=trackCollection.product()->begin();
			
      hphi->Fill((*ibeg).outerPhi());
      hnhit->Fill((*ibeg).recHitsSize());
      hEventTrackNumber->Fill(tracks->size());

      std::cout << " N hits on Track = " << (*ibeg).recHitsSize() << std::endl;
			
      std::vector<std::pair<const TrackingRecHit *,float> > tmphitangle=anglefinder_->findtrackangle((*(*seedcoll).begin()),tracks->front());
      std::vector<std::pair<const TrackingRecHit *,float> >::iterator tmpiter;
	
      trackhits[&(*ibeg)] = tmphitangle;
      trackhitsXZ[&(*ibeg)] = anglefinder_->getXZHitAngle();
      trackhitsYZ[&(*ibeg)] = anglefinder_->getYZHitAngle();
      oLclDirs[&(*ibeg)] = anglefinder_->getLocalDir();
      oGlbDirs[&(*ibeg)] = anglefinder_->getGlobalDir();
	
      for(tmpiter=tmphitangle.begin();tmpiter!=tmphitangle.end();++tmpiter){
	hitcounter+=1;
      }
    }
  }
  // -------------------------------------------------------------------------------------------
  // -------------------------------------------------------------------------------------------



  // Get SiStripClusterInfos
  edm::Handle<edm::DetSetVector<SiStripClusterInfo> > oDSVClusterInfos;
  e.getByLabel( "siStripClusterInfoProducer", oDSVClusterInfos);

  // Get SiStripDigis
  std::cout << "oSiStripDigisProdInstNama_.size() = " << oSiStripDigisProdInstName_.size() << std::endl;
  edm::Handle<edm::DetSetVector<SiStripDigi> > oDSVDigis;
  if( oSiStripDigisProdInstName_.size()) {
    std::cout << "if" << std::endl;
    e.getByLabel( oSiStripDigisLabel_.c_str(), oSiStripDigisProdInstName_.c_str(), oDSVDigis);
  } else {
    std::cout << "else" << std::endl;
    e.getByLabel( oSiStripDigisLabel_.c_str(), oDSVDigis);
  }

  std::cout << "oDSVDigis->size() = " << oDSVDigis->size() << std::endl;

  // Cluster loop
  std::map<uint32_t, int> oProcessedClusters;

  int nTrackClusters = 0;

  if(trackhits.size()!=0){

    std::cout<< "TrackingHit Map size = " << trackhits.size() << std::endl;
       
    trackhitmap::iterator mapiter;
    
    for(mapiter = trackhits.begin(); mapiter != trackhits.end(); ++mapiter){
          
      momentum=-99;
      pt=-99;
      charge=-99;
           
      momentum = (*mapiter).first->p();
      pt = (*mapiter).first->pt();
      eta      = mapiter->first->eta();
      phi      = mapiter->first->phi();
      chi2     = mapiter->first->chi2();
      ndof     = mapiter->first->ndof();
      normchi2 = mapiter->first->normalizedChi2();
      hitspertrack = mapiter->first->recHitsSize();
      charge = (*mapiter).first->charge();
     
      poTrackTree->Fill();
                
      std::vector<std::pair<const TrackingRecHit *,float> > hitangle = (*mapiter).second;
      std::vector<std::pair<const TrackingRecHit *,float> >::iterator hitsiter;

      if(hitangle.size()!=0){
	// create reference to YZAngles
	TrackLocalAngleTIF::HitAngleAssociation &roHitAngleAssocXZ = trackhitsXZ[mapiter->first];
	TrackLocalAngleTIF::HitAngleAssociation &roHitAngleAssocYZ = trackhitsYZ[mapiter->first];

	// Remove candidates
	// -----------------
	//	TrackLocalAngleTIF::HitLclDirAssociation &roLclDirAssoc    = oLclDirs[mapiter->first];
	//	TrackLocalAngleTIF::HitGlbDirAssociation &roGlbDirAssoc    = oGlbDirs[mapiter->first];
	// -----------------

	int nHitNum = 0;
	for(hitsiter=hitangle.begin();hitsiter!=hitangle.end();++hitsiter){

	  TrackLocalAngleTIF::HitAngleAssociation::reference hitsrefXZ = roHitAngleAssocXZ[nHitNum];
	  TrackLocalAngleTIF::HitAngleAssociation::reference hitsrefYZ = roHitAngleAssocYZ[nHitNum];

	  // Remove candidates
	  // -----------------
	  //	  TrackLocalAngleTIF::HitLclDirAssociation::reference roLclDir = roLclDirAssoc[nHitNum];
	  //	  TrackLocalAngleTIF::HitGlbDirAssociation::reference roGlbDir = roGlbDirAssoc[nHitNum];
	  // -----------------
	  ++nHitNum;
        
	  module=-99;
	  type=-99;
	  layer=-99;
	  wheel=-99;
	  string=-99;
	  size=-99;
	  extint=-99;
	  bwfw=-99;
	  rod=-99;
	  angle=-9999;
	  tk_phi=-9999;
	  tk_theta=-9999;
	  stereocorrection=-9999;
	  localmagfield=-99;
	  sign = -99;
	  monostereo=-99;
	  clusternoise=-99;
	  clusterbarycenter=-99;
	  clusterseednoise=-99;

	  const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(hitsiter->first);
	  dLclX = hit->localPosition().x(); 
	  dLclY = hit->localPosition().y(); 
	  dLclZ = hit->localPosition().z(); 

	  GlobalPoint oRecHitGlobalPos = tracker->idToDet( hit->geographicalId())->toGlobal( hit->localPosition());
	  dGlbX = oRecHitGlobalPos.x();
	  dGlbY = oRecHitGlobalPos.y();
	  dGlbZ = oRecHitGlobalPos.z();

	  const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=hit->cluster();
	  std::cout << "after third detsetvector (clusters)" << std::endl;



	  std::vector<SiStripDigi> oDigis;
	  std::cout << "cluster->geographicalId() = " << cluster->geographicalId() << std::endl;
	  oDigis = oDSVDigis->operator[]( cluster->geographicalId()).data;



	  //use SiStripClusterInfo to fill the size. Look later on
 
	  //std::cout << "size = (cluster->amplitudes()).size() = " << (cluster->amplitudes()).size() << std::endl;
	  //std::cout << "Cluster size = " << amplitudes.size() << std::endl;

	  //      size=(cluster->amplitudes()).size();

	  /*
	    clustereta = getClusterEta( cluster->amplitudes(),
	    cluster->firstStrip(),
	    oDigis);
	    clustercrosstalk = getClusterCrossTalk( cluster->amplitudes(),
	    cluster->firstStrip(),
	    oDigis);
	  */




	  std::cout << "cluster->geographicalId() = " << cluster->geographicalId() << std::endl;
	  std::vector<SiStripClusterInfo> oClusterInfos = 
	    oDSVClusterInfos->operator[]( cluster->geographicalId()).data;





	  for( std::vector<SiStripClusterInfo>::iterator oIter = 
		 oClusterInfos.begin();
	       oIter != oClusterInfos.end();
	       ++oIter) {
	
	    if( oIter->firstStrip() == cluster->firstStrip()) {


	      // Check if it was already processed and take some action: flag or don't write it.
	      // if use a flag: you can exclude it from disributions of all clusters, track cluters...
	      // if don't write: you don't have the information when looking at single tracks,
	      // for example you want to look at all the clusters which belong to the second track in a given event...
	      //	if( oProcessedClusters[oDSVIter->id] != oIter->firstStrip()) {









	      // ClusterInfo matched given cluster
	      clusterpos    = oIter->position();
	      clusterchg	= oIter->charge();

	      std::cout << "Cluster Info size = " << oIter->stripAmplitudes().size() << " or  width " << oIter->width() << std::endl;

	      std::cout << "Cluster Info Charge = " << clusterchg << std::endl;


	      //fill here using clusterinfo thing
	      clustereta = getClusterEta( oIter->stripAmplitudes(),
					  oIter->firstStrip(),
					  oDigis);
	      clustercrosstalk = getClusterCrossTalk( oIter->stripAmplitudes(),
						      oIter->firstStrip(),
						      oDigis);

	      size = int(oIter->width());
	      //==============================

	      clusterchgl   = oIter->chargeL();
	      clusterchgr   = oIter->chargeR();
	      clusternoise  = oIter->noise();
	      clusterbarycenter = cluster->barycenter();
	      clusterseednoise = oIter->stripNoises().operator[]( ( int) cluster->barycenter() - cluster->firstStrip());
	      clustermaxchg = oIter->maxCharge();
	      bTrack = true;

	      poClusterChargeTH1F->Fill( clusterchg);

	      // Mark Current ClusterInfo as processed
	      // [Note: One Module might hold several clusters (!)]
	      oProcessedClusters[cluster->geographicalId()] = oIter->firstStrip();

	      break;
	    } else {
	      // ClusterInfo was found but didn't match given cluster
	    }
	  } // end clusterInfo loop

	  StripSubdetector detid=(StripSubdetector)hit->geographicalId();

	  type = detid.subdetId();
	  std::cout << "type = " << type << std::endl;

	  module = (hit->geographicalId()).rawId();
      
	  angle=hitsiter->second;
	  tk_phi=hitsrefXZ.second;
	  tk_theta=hitsrefYZ.second;
	  float angleXZ=hitsrefXZ.second;
	  float angleYZ=hitsrefYZ.second;

	  monostereo=detid.stereo();
	    
	  //Local Magnetic Field
	    
	  const GeomDet *geomdet = tracker->idToDet(hit->geographicalId());
	  LocalPoint localp(0,0,0);
	  const GlobalPoint globalp = (geomdet->surface()).toGlobal(localp);
	  GlobalVector globalmagdir = magfield->inTesla(globalp);
	  localmagdir = (geomdet->surface()).toLocal(globalmagdir);
	  localmagfield = localmagdir.mag();
	    
	  if(localmagfield != 0.){
	      
	    // ---------------
	    // CHECK THIS PART
	    // ---------------
	    //Sign correction for TIB and TOB
	      
	    if((detid.subdetId() == int (StripSubdetector::TIB)) || (detid.subdetId() == int (StripSubdetector::TOB))){
	      
	      LocalVector ylocal(0,1,0);
	      
	      float normprojection = (localmagdir * ylocal)/(localmagfield);
	      
	      if(normprojection>0){sign = 1;}
	      if(normprojection<0){sign = -1;}
		
	      //Stereocorrection applied in TrackLocalAngleTIF
		
	      if((detid.stereo()==1) && (normprojection == 0.)){
		LogDebug("TIFNtupleMaker::analyze")<<"Error: TIB|TOB YBprojection = 0";
	      }
		
	      if((detid.stereo()==1) && (normprojection != 0.)){
		stereocorrection = 1/normprojection;
		stereocorrection*=sign;
		  
		float tg = tan((angle*TMath::Pi())/180);
		tg*=stereocorrection;
		angle = atan(tg)*180/TMath::Pi();
		  
		tg = tan( angleXZ);
		tg*=stereocorrection;
		angleXZ = atan( tg);
		  
		tg = tan( angleYZ);
		tg*=stereocorrection;
		angleYZ = atan( tg);
	      }
		
	      angle   *= sign;
	      angleXZ *= sign;
	      angleYZ *= sign;       
	    }
	  } // end if localmagfield!=0
	    // -------------------
	    // -------------------

	    //Filling histograms
	    
	  if(conf_.getParameter<bool>("SINGLE_DETECTORS")) {
	    histos[module]->Fill(angle,size);
	      
	    LogDebug("TIFNtupleMaker::analyze")<<"Module histogram filled";
	  }
	    
	  //Summary histograms
	    
	  if(detid.subdetId() == int (StripSubdetector::TIB)){
	    TIBDetId TIBid=TIBDetId(hit->geographicalId());
	      
	    extint = TIBid.string()[1];
	    string = TIBid.string()[2];
	    bwfw= TIBid.string()[0];
	    layer = TIBid.layer();
	      
	    if(layer == 1){
	      histos[1]->Fill(angle,size);
	      if(TIBid.stereo()==0){
		htaTIBL1mono->Fill(angle);}
	      if(TIBid.stereo()==1){
		htaTIBL1stereo->Fill(angle);}
		
	      if(TIBid.string()[1]==0){//int
		histos[2]->Fill(angle,size);
	      }
	      if(TIBid.string()[1]==1){//ext
		histos[3]->Fill(angle,size);
	      }
	    }
	      
	    if(layer == 2){
	      histos[4]->Fill(angle,size);
	      if(TIBid.stereo()==0){
		htaTIBL2mono->Fill(angle);}
	      if(TIBid.stereo()==1){
		htaTIBL2stereo->Fill(angle);}
		
	      if(TIBid.string()[1]==0){//int
		histos[5]->Fill(angle,size);
	      }
	      if(TIBid.string()[1]==1){//ext
		histos[6]->Fill(angle,size);
	      }
	    }
	      
	    if(layer == 3){
	      std::cout << "angle = " << angle << std::endl;
	      std::cout << "size = " << size << std::endl;
	      histos[7]->Fill(angle,size);
	      std::cout << "non stereo layer TIBid.stereo() = " << TIBid.stereo() << std::endl;
	      htaTIBL3->Fill(angle);
	      if(TIBid.string()[1]==0){//int
		histos[8]->Fill(angle,size);
	      }
	      if(TIBid.string()[1]==1){//ext
		histos[9]->Fill(angle,size);
	      }
	    }
	      
	    if(layer == 4){
	      histos[10]->Fill(angle,size);
	      htaTIBL4->Fill(angle);
	      if(TIBid.string()[1]==0){//int
		histos[11]->Fill(angle,size);
	      }
	      if(TIBid.string()[1]==1){//ext
		histos[12]->Fill(angle,size);
	      }
	    }
	  }

	  if(conf_.getParameter<bool>("TOB_ON")) {
	    if(detid.subdetId() == int (StripSubdetector::TOB)){
	      TOBDetId TOBid=TOBDetId(hit->geographicalId());
	      
	      layer = TOBid.layer();
	      rod = TOBid.rod()[1];
	      bwfw = TOBid.rod()[0];
	    
	      if(layer == 1){
		histos[13]->Fill(angle,size);		
		if(TOBid.stereo()==0){//mono
		  htaTOBL1mono->Fill(angle);
		}
		if(TOBid.stereo()==1){//stereo
		  htaTOBL1stereo->Fill(angle);
		}
	      }
	      if(layer == 2){
		histos[14]->Fill(angle,size);		
		if(TOBid.stereo()==0){//mono
		  htaTOBL2mono->Fill(angle);
		}
		if(TOBid.stereo()==1){//stereo
		  htaTOBL2stereo->Fill(angle);
		}
	      }
	      if(layer == 3){
		histos[15]->Fill(angle,size);
		htaTOBL3->Fill(angle);}
	    
	      if(layer == 4){
		histos[16]->Fill(angle,size);
		htaTOBL4->Fill(angle);}
	    
	      if(layer == 5){
		histos[17]->Fill(angle,size);
		htaTOBL5->Fill(angle);}
	    
	      if(layer == 6){
		histos[18]->Fill(angle,size);
		htaTOBL6->Fill(angle);}
	    }
	  }

	  if(conf_.getParameter<bool>("TID_ON")) {
	    if(detid.subdetId() == int (StripSubdetector::TID)){
	      TIDDetId TIDid=TIDDetId(hit->geographicalId());
	      bwfw = TIDid.module()[0];
	      wheel = TIDid.wheel();
	      //aggiungere plots per il tid
	    }
	  }

	  if(conf_.getParameter<bool>("TEC_ON")) {
	    if(detid.subdetId() == int (StripSubdetector::TEC)){
	      TECDetId TECid=TECDetId(hit->geographicalId());
	      bwfw = TECid.petal()[0];
	      wheel = TECid.wheel();
	    }
	  }

	  const GeomDetUnit * stripdet=(const GeomDetUnit*)tracker->idToDetUnit(detid);
      
	  const StripTopology& topol=(StripTopology&)stripdet->topology();
	  
	  float thickness=stripdet->specificSurface().bounds().thickness();
	  
	  float proj=tan(angle)*thickness/topol.pitch();
	  
	  //Filling WidthvsTrackProjection histogram
	  
	  hwvst->Fill(proj,size);
	  
	  //Filling Tree
	  
	  TIFNtupleMakerTree->Fill();

	  LogDebug("TIFNtupleMaker::analyze")<<"Tree Filled";
      
	} // end hitsiter loop
	nTrackClusters += hitangle.size();
      } // end if hitangle.size != 0

    } // trackhits loop (on the map)
  } // if trackhits map.size != 0

  std::map<SiSubDet, std::map<unsigned char, int> > oClustersPerLayer;
  int nTotClusters = nTrackClusters;

  // Now work out all Cluster that were left unprocessed thus do not belong to
  // tracks

  // Loop over modules
  for( edm::DetSetVector<SiStripClusterInfo>::const_iterator oDSVIter = oDSVClusterInfos->begin();
       oDSVIter != oDSVClusterInfos->end();
       ++oDSVIter) {
    std::cout << "inside loop over modules" << std::endl;

    StripSubdetector oStripSubdet( oDSVIter->id);
    unsigned char ucLayer;
    switch( oStripSubdet.subdetId()) {
    case StripSubdetector::TIB:
      {
	if(conf_.getParameter<bool>("TIB_ON")){
	  TIBDetId oTIBDetId( oStripSubdet);
	
	  ucLayer  = oTIBDetId.layer();
	}
	break;
      }
    case StripSubdetector::TID:
      if(conf_.getParameter<bool>("TID_ON")){
      }
      break;
    case StripSubdetector::TOB:
      {
	if(conf_.getParameter<bool>("TOB_ON")) {
	  TOBDetId oTOBDetId( oStripSubdet);
	  
	  ucLayer = oTOBDetId.layer();
	}
	break;
      }
    case StripSubdetector::TEC:
      if(conf_.getParameter<bool>("TEC_ON")){
      }
      break;
    }
    oClustersPerLayer[SiSubDet( oStripSubdet.subdetId())][ucLayer] += oDSVIter->data.size();

    if( oProcessedClusters.end() ==
	oProcessedClusters.find( oDSVIter->id)) {

      // Extract ClusterInfos collection for given module
      std::vector<SiStripClusterInfo> oClusterInfos = oDSVIter->data;
      std::vector<SiStripDigi> oDigis;
      oDigis = oDSVDigis->operator[]( oDSVIter->id).data;

      // Work in progress
      // ----------------
      // Different detectors -> different counters,
      // or else we cannot distinguish in simulation
      //       if(conf_.getParameter<bool>("TIB_ON") && StripSubdetector::TIB == oStripSubdet.subdetId()){
      // 	nTotClustersTIB += oClusterInfos.size();
      //       }
      //       if(conf_.getParameter<bool>("TOB_ON") && StripSubdetector::TOB == oStripSubdet.subdetId()){
      // 	nTotClustersTOB += oClusterInfos.size();
      //       }
      //       if(conf_.getParameter<bool>("TID_ON") && StripSubdetector::TID == oStripSubdet.subdetId()){
      // 	nTotClustersTID += oClusterInfos.size();
      //       }
      //       if(conf_.getParameter<bool>("TEC_ON") && StripSubdetector::TEC == oStripSubdet.subdetId()){
      // 	nTotClustersTEC += oClusterInfos.size();
      //       }
      // ----------------

      // Loop over ClusterInfos collection
      for( std::vector<SiStripClusterInfo>::iterator oIter = 
	     oClusterInfos.begin();
	   oIter != oClusterInfos.end();
	   ++oIter) {
	
	// Check if given ClusterInfo was already processed
	if( oProcessedClusters[oDSVIter->id] !=
	    oIter->firstStrip()) {

	  // Set default values to the rest of the variables
	  pt	       = -99;
	  eta	       = -99;
	  phi	       = -99;
	  charge       = -99;
	  momentum     = -99;
	  normchi2     = -99;
	  hitspertrack = -99;
	  dLclX	       = -99;
	  dLclY	       = -99;
	  dLclZ	       = -99;
	  dGlbX	       = -99;
	  dGlbY	       = -99;
	  dGlbZ	       = -99;

	  rod		     = -99;	  // Only for TOB
	  sign	     = -99;	  // Not needed
	  wheel	     = -99;	  // Only for TID, TEC
	  angle	     = -9999; // Only for track
	  layer	     = -99;	  // There is no layer for TID and TEC
	  string	     = -99;	  // String exists only for TIB
	  extint	     = -99;	  // Olny for TIB
	  localmagfield    = -99;	  // Not needed
	  stereocorrection = -9999; // Only for track
	  clusterbarycenter = -99;
	  clusterseednoise = -99;

	  StripSubdetector oStripSubdet( oIter->geographicalId());
	  type	   = oStripSubdet.subdetId();
	  std::cout << "type = " << type << std::endl;
	  module   = oStripSubdet.rawId();
	  monostereo = oStripSubdet.stereo();

	  switch( type) {
	  case StripSubdetector::TIB:
	    {
	      if(conf_.getParameter<bool>("TIB_ON")){
		TIBDetId oTIBDetId( oStripSubdet);

		bwfw   = oTIBDetId.string()[0];
		layer  = oTIBDetId.layer();
		extint = oTIBDetId.string()[1];
		string = oTIBDetId.string()[2];
	      }
	      break;
	    }
	  case StripSubdetector::TID:
	    {
	      if(conf_.getParameter<bool>("TID_ON")){
		TIDDetId oTIDDetId( oStripSubdet);

		bwfw  = oTIDDetId.module()[0];
		wheel = oTIDDetId.wheel();
	      }
	      break;
	    }
	  case StripSubdetector::TOB:
	    {
	      if(conf_.getParameter<bool>("TOB_ON")) {
		TOBDetId oTOBDetId( oStripSubdet);

		rod   = oTOBDetId.rod()[1];
		bwfw  = oTOBDetId.rod()[0];
		std::cout << "rod = " << rod << std::endl;
		std::cout << "bwfw = " << bwfw << std::endl;
		layer = oTOBDetId.layer();
		std::cout << "layer = " << layer << std::endl;
	      }
	      break;
	    }
	  case StripSubdetector::TEC:
	    {
	      if(conf_.getParameter<bool>("TEC_ON")) {
		TECDetId oTECDetId( oStripSubdet);

		bwfw  = oTECDetId.petal()[0];
		wheel = oTECDetId.wheel();
	      }
	      break;
	    }
	  }

	  //	  std::cout << "cluster width = " << oIter->width() << std::endl;
	  size	      = int(oIter->width()); // cluster width
	  std::cout << "size = " << size << std::endl;


	  // ClusterInfo was not processed yet
	  clusterpos    = oIter->position();
	  clusterchg    = oIter->charge();
	  clusterchgl   = oIter->chargeL();
	  clusterchgr   = oIter->chargeR();
	  clusternoise  = oIter->noise();
	  clustermaxchg = oIter->maxCharge();
	  clustereta    = getClusterEta( oIter->stripAmplitudes(), 
					 oIter->firstStrip(),
					 oDigis);
	  clustercrosstalk = getClusterCrossTalk( oIter->stripAmplitudes(),
						  oIter->firstStrip(),
						  oDigis);

	  bTrack = false;

	  TIFNtupleMakerTree->Fill();
	}
      }
    }
  }

  // Work in progress
  // ----------------
  //   if (conf_.getParameter<bool>("TIB_ON")) {
  //     nTotClusters += nTotClustersTIB;
  //   }
  //   if (conf_.getParameter<bool>("TOB_ON")) {
  //     nTotClusters += nTotClustersTOB;
  //   }
  //   if (conf_.getParameter<bool>("TID_ON")) {
  //     nTotClusters += nTotClustersTID;
  //   }
  //   if (conf_.getParameter<bool>("TEC_ON")) {
  //     nTotClusters += nTotClustersTEC;
  //   }
  //-----------------

  oGlobalPlots[0]->Fill( ( 1.0 * nTrackClusters) / nTotClusters);

  if(conf_.getParameter<bool>("TIB_ON")){
    std::cout << "oClustersPerLayer[StripSubdetector::TIB][1] = " << oClustersPerLayer[StripSubdetector::TIB][1] << std::endl;
    std::cout << "oClustersPerLayer[StripSubdetector::TIB][2] = " << oClustersPerLayer[StripSubdetector::TIB][2] << std::endl;
    std::cout << "oClustersPerLayer[StripSubdetector::TIB][3] = " << oClustersPerLayer[StripSubdetector::TIB][3] << std::endl;
    std::cout << "oClustersPerLayer[StripSubdetector::TIB][4] = " << oClustersPerLayer[StripSubdetector::TIB][4] << std::endl;
    //     if( 0 < oClustersPerLayer[StripSubdetector::TIB][1])
    //       oDetPlots[StripSubdetector::TIB][0]->Fill( oClustersPerLayer[StripSubdetector::TIB][1]);

    //     if( 0 < oClustersPerLayer[StripSubdetector::TIB][2])
    //       oDetPlots[StripSubdetector::TIB][1]->Fill( oClustersPerLayer[StripSubdetector::TIB][2]);

    //     if( 0 < oClustersPerLayer[StripSubdetector::TIB][3])
    //       oDetPlots[StripSubdetector::TIB][2]->Fill( oClustersPerLayer[StripSubdetector::TIB][3]);

    //     if( 0 < oClustersPerLayer[StripSubdetector::TIB][4])
    //       oDetPlots[StripSubdetector::TIB][3]->Fill( oClustersPerLayer[StripSubdetector::TIB][4]);
  }
 
//   if(conf_.getParameter<bool>("TOB_ON")) {
//     if( 0 < oClustersPerLayer[StripSubdetector::TOB][3])
//       oDetPlots[StripSubdetector::TOB][0]->Fill( oClustersPerLayer[StripSubdetector::TOB][3]);
//     if( 0 < oClustersPerLayer[StripSubdetector::TOB][4])
//       oDetPlots[StripSubdetector::TOB][1]->Fill( oClustersPerLayer[StripSubdetector::TOB][4]);
//     if( 0 < oClustersPerLayer[StripSubdetector::TOB][3])
//       oDetPlots[StripSubdetector::TOB][2]->Fill( oClustersPerLayer[StripSubdetector::TOB][5]);
//     if( 0 < oClustersPerLayer[StripSubdetector::TOB][4])
//       oDetPlots[StripSubdetector::TOB][3]->Fill( oClustersPerLayer[StripSubdetector::TOB][6]);
//   }

  //sistemare e add TID

  std::cout << "TID and TEC here" << std::endl;

  // Fill tracks number
  numberoftracks = tracks->size();
  numberofclusters = nTotClusters;
  numberoftkclusters = nTrackClusters;
  numberofnontkclusters = nTotClusters - nTrackClusters;
  poTrackNum->Fill();

} // end analyze

//EndJob

void TIFNtupleMaker::endJob(){

  std::vector<DetId>::iterator Iditer;
  
  //Histograms fit
  
  std::cout << "endJob" << std::endl;
  int histonum = Detvector.size();

  if(conf_.getParameter<bool>("SINGLE_DETECTORS")) {

    std::cout << "histonum = " << histonum << std::endl;

    for(Iditer=Detvector.begin(); Iditer!=Detvector.end(); ++Iditer){
    
      fitfunc->SetParameters(0, 0, 1);

      histos[Iditer->rawId()]->Fit("fitfunc","E","",-20, 20);
    
      TF1 *fitfunction = histos[Iditer->rawId()]->GetFunction("fitfunc");

      fits[Iditer->rawId()] = new histofit;
    
      fits[Iditer->rawId()]->chi2 = fitfunction->GetChisquare();
      fits[Iditer->rawId()]->ndf  = fitfunction->GetNDF();
      fits[Iditer->rawId()]->p0   = fitfunction->GetParameter(0);
      fits[Iditer->rawId()]->p1   = fitfunction->GetParameter(1);
      fits[Iditer->rawId()]->p2   = fitfunction->GetParameter(2);
      fits[Iditer->rawId()]->errp0   = fitfunction->GetParError(0);
      fits[Iditer->rawId()]->errp1   = fitfunction->GetParError(1);
      fits[Iditer->rawId()]->errp2   = fitfunction->GetParError(2);
      fits[Iditer->rawId()]->min     = fitfunction->Eval(fits[Iditer->rawId()]->p0);    
    }
  } // SINGLE_DETECTORS

  int n;
  int nmax = 19;

  for(n=1; n<nmax; ++n){
    
    fitfunc->SetParameters(0, 0, 1);
    histos[n]->Fit("fitfunc","E","",-20, 20);
    
    TF1 *fitfunction = histos[n]->GetFunction("fitfunc");
    
    fits[n] = new histofit;

    fits[n]->chi2 = fitfunction->GetChisquare();
    fits[n]->ndf  = fitfunction->GetNDF();
    fits[n]->p0   = fitfunction->GetParameter(0);
    fits[n]->p1   = fitfunction->GetParameter(1);
    fits[n]->p2   = fitfunction->GetParameter(2);
    fits[n]->errp0   = fitfunction->GetParError(0);
    fits[n]->errp1   = fitfunction->GetParError(1);
    fits[n]->errp2   = fitfunction->GetParError(2);
    fits[n]->min     = fitfunction->Eval(fits[n]->p0);
    std::cout <<  "fits["<<n<<"]->chi2 = " << fits[n]->chi2  << std::endl;
    std::cout <<  "fits["<<n<<"]->ndf = " << fits[n]->ndf    << std::endl;
    std::cout <<  "fits["<<n<<"]->p0 = " << fits[n]->p0      << std::endl;
    std::cout <<  "fits["<<n<<"]->p1 = " << fits[n]->p1      << std::endl;
    std::cout <<  "fits["<<n<<"]->p2 = " << fits[n]->p2      << std::endl;
    std::cout <<  "fits["<<n<<"]->errp0 = " << fits[n]->errp0  << std::endl;
    std::cout <<  "fits["<<n<<"]->errp1 = " << fits[n]->errp1  << std::endl;
    std::cout <<  "fits["<<n<<"]->errp2 = " << fits[n]->errp2  << std::endl;
    std::cout <<  "fits["<<n<<"]->min = " << fits[n]->min  << std::endl;
  
  }

  //File with fit parameters  
  
  ofstream fit;
  fit.open("fit.txt");
  
  std::cout << "before some fits" << std::endl;

  fit<<">>> TOTAL EVENT = "<<eventcounter<<endl;
  fit<<">>> NUMBER OF RECHITS = "<<hitcounter<<endl;
  fit<<">>> NUMBER OF TRACKS = "<<trackcounter<<endl<<endl;

  if(conf_.getParameter<bool>("SINGLE_DETECTORS")){
    fit<<">>> NUMBER OF DETECTOR HISTOGRAMS = "<<histonum<<endl;
    fit<<">>> NUMBER OF MONO SINGLE SIDED DETECTORS = "<<monosscounter<<endl;
    fit<<">>> NUMBER OF MONO DOUBLE SIDED DETECTORS = "<<monodscounter<<endl;
    fit<<">>> NUMBER OF STEREO DETECTORS = "<<stereocounter<<endl<<endl;
  }

  if(conf_.getParameter<bool>("TIB_ON")){
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 1 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[9]->chi2)/(fits[9]->ndf)<<endl;
    fit<<"NdF        = "<<fits[9]->ndf<<endl;
    fit<<"p0 = "<<fits[9]->p0<<"     err p0 = "<<fits[9]->errp0<<endl;
    fit<<"p1 = "<<fits[9]->p1<<"     err p1 = "<<fits[9]->errp1<<endl;
    fit<<"p2 = "<<fits[9]->p2<<"     err p2 = "<<fits[9]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[9]->p0<<"  +-  "<<fits[9]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[9]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 1 INT -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[10]->chi2)/(fits[10]->ndf)<<endl;
    fit<<"NdF        = "<<fits[10]->ndf<<endl;
    fit<<"p0 = "<<fits[10]->p0<<"     err p0 = "<<fits[10]->errp0<<endl;
    fit<<"p1 = "<<fits[10]->p1<<"     err p1 = "<<fits[10]->errp1<<endl;
    fit<<"p2 = "<<fits[10]->p2<<"     err p2 = "<<fits[10]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[10]->p0<<"  +-  "<<fits[10]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[10]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 1 EXT -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[11]->chi2)/(fits[11]->ndf)<<endl;
    fit<<"NdF        = "<<fits[11]->ndf<<endl;
    fit<<"p0 = "<<fits[11]->p0<<"     err p0 = "<<fits[11]->errp0<<endl;
    fit<<"p1 = "<<fits[11]->p1<<"     err p1 = "<<fits[11]->errp1<<endl;
    fit<<"p2 = "<<fits[11]->p2<<"     err p2 = "<<fits[11]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[11]->p0<<"  +-  "<<fits[11]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[11]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 2 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[1]->chi2)/(fits[1]->ndf)<<endl;
    fit<<"NdF        = "<<fits[1]->ndf<<endl;
    fit<<"p0 = "<<fits[1]->p0<<"     err p0 = "<<fits[1]->errp0<<endl;
    fit<<"p1 = "<<fits[1]->p1<<"     err p1 = "<<fits[1]->errp1<<endl;
    fit<<"p2 = "<<fits[1]->p2<<"     err p2 = "<<fits[1]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[1]->p0<<"  +-  "<<fits[1]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[1]->min<<endl<<endl;
  
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 2 INT -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[2]->chi2)/(fits[2]->ndf)<<endl;
    fit<<"NdF        = "<<fits[2]->ndf<<endl;
    fit<<"p0 = "<<fits[2]->p0<<"     err p0 = "<<fits[2]->errp0<<endl;
    fit<<"p1 = "<<fits[2]->p1<<"     err p1 = "<<fits[2]->errp1<<endl;
    fit<<"p2 = "<<fits[2]->p2<<"     err p2 = "<<fits[2]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[2]->p0<<"  +-  "<<fits[2]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[2]->min<<endl<<endl;
  
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 2 EXT -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[3]->chi2)/(fits[3]->ndf)<<endl;
    fit<<"NdF        = "<<fits[3]->ndf<<endl;
    fit<<"p0 = "<<fits[3]->p0<<"     err p0 = "<<fits[3]->errp0<<endl;
    fit<<"p1 = "<<fits[3]->p1<<"     err p1 = "<<fits[3]->errp1<<endl;
    fit<<"p2 = "<<fits[3]->p2<<"     err p2 = "<<fits[3]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[3]->p0<<"  +-  "<<fits[3]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[3]->min<<endl<<endl;
  
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 3 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[4]->chi2)/(fits[4]->ndf)<<endl;
    fit<<"NdF        = "<<fits[4]->ndf<<endl;
    fit<<"p0 = "<<fits[4]->p0<<"     err p0 = "<<fits[4]->errp0<<endl;
    fit<<"p1 = "<<fits[4]->p1<<"     err p1 = "<<fits[4]->errp1<<endl;
    fit<<"p2 = "<<fits[4]->p2<<"     err p2 = "<<fits[4]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[4]->p0<<"  +-  "<<fits[4]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[4]->min<<endl<<endl;
  
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 3 INT -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[5]->chi2)/(fits[5]->ndf)<<endl;
    fit<<"NdF        = "<<fits[5]->ndf<<endl;
    fit<<"p0 = "<<fits[5]->p0<<"     err p0 = "<<fits[5]->errp0<<endl;
    fit<<"p1 = "<<fits[5]->p1<<"     err p1 = "<<fits[5]->errp1<<endl;
    fit<<"p2 = "<<fits[5]->p2<<"     err p2 = "<<fits[5]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[5]->p0<<"  +-  "<<fits[5]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[5]->min<<endl<<endl;
  
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 3 EXT -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[6]->chi2)/(fits[6]->ndf)<<endl;
    fit<<"NdF        = "<<fits[6]->ndf<<endl;
    fit<<"p0 = "<<fits[6]->p0<<"     err p0 = "<<fits[6]->errp0<<endl;
    fit<<"p1 = "<<fits[6]->p1<<"     err p1 = "<<fits[6]->errp1<<endl;
    fit<<"p2 = "<<fits[6]->p2<<"     err p2 = "<<fits[6]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[6]->p0<<"  +-  "<<fits[6]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[6]->min<<endl<<endl;
  
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 4 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[12]->chi2)/(fits[12]->ndf)<<endl;
    fit<<"NdF        = "<<fits[12]->ndf<<endl;
    fit<<"p0 = "<<fits[12]->p0<<"     err p0 = "<<fits[12]->errp0<<endl;
    fit<<"p1 = "<<fits[12]->p1<<"     err p1 = "<<fits[12]->errp1<<endl;
    fit<<"p2 = "<<fits[12]->p2<<"     err p2 = "<<fits[12]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[12]->p0<<"  +-  "<<fits[12]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[12]->min<<endl<<endl;
  
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 4 INT -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[13]->chi2)/(fits[13]->ndf)<<endl;
    fit<<"NdF        = "<<fits[13]->ndf<<endl;
    fit<<"p0 = "<<fits[13]->p0<<"     err p0 = "<<fits[13]->errp0<<endl;
    fit<<"p1 = "<<fits[13]->p1<<"     err p1 = "<<fits[13]->errp1<<endl;
    fit<<"p2 = "<<fits[13]->p2<<"     err p2 = "<<fits[13]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[13]->p0<<"  +-  "<<fits[13]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[13]->min<<endl<<endl;
  
    fit<<endl<<"--------------------------- SUMMARY FIT: TIB LAYER 4 EXT -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[14]->chi2)/(fits[14]->ndf)<<endl;
    fit<<"NdF        = "<<fits[14]->ndf<<endl;
    fit<<"p0 = "<<fits[14]->p0<<"     err p0 = "<<fits[14]->errp0<<endl;
    fit<<"p1 = "<<fits[14]->p1<<"     err p1 = "<<fits[14]->errp1<<endl;
    fit<<"p2 = "<<fits[14]->p2<<"     err p2 = "<<fits[14]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[14]->p0<<"  +-  "<<fits[14]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[14]->min<<endl<<endl;    
  }

  if(conf_.getParameter<bool>("TOB_ON")) {
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 1 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[15]->chi2)/(fits[15]->ndf)<<endl;
    fit<<"NdF        = "<<fits[15]->ndf<<endl;
    fit<<"p0 = "<<fits[15]->p0<<"     err p0 = "<<fits[15]->errp0<<endl;
    fit<<"p1 = "<<fits[15]->p1<<"     err p1 = "<<fits[15]->errp1<<endl;
    fit<<"p2 = "<<fits[15]->p2<<"     err p2 = "<<fits[15]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[15]->p0<<"  +-  "<<fits[15]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[15]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 2 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[16]->chi2)/(fits[16]->ndf)<<endl;
    fit<<"NdF        = "<<fits[16]->ndf<<endl;
    fit<<"p0 = "<<fits[16]->p0<<"     err p0 = "<<fits[16]->errp0<<endl;
    fit<<"p1 = "<<fits[16]->p1<<"     err p1 = "<<fits[16]->errp1<<endl;
    fit<<"p2 = "<<fits[16]->p2<<"     err p2 = "<<fits[16]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[16]->p0<<"  +-  "<<fits[16]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[16]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 3 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[7]->chi2)/(fits[7]->ndf)<<endl;
    fit<<"NdF        = "<<fits[7]->ndf<<endl;
    fit<<"p0 = "<<fits[7]->p0<<"     err p0 = "<<fits[7]->errp0<<endl;
    fit<<"p1 = "<<fits[7]->p1<<"     err p1 = "<<fits[7]->errp1<<endl;
    fit<<"p2 = "<<fits[7]->p2<<"     err p2 = "<<fits[7]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[7]->p0<<"  +-  "<<fits[7]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[7]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 4 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[8]->chi2)/(fits[8]->ndf)<<endl;
    fit<<"NdF        = "<<fits[8]->ndf<<endl;
    fit<<"p0 = "<<fits[8]->p0<<"     err p0 = "<<fits[8]->errp0<<endl;
    fit<<"p1 = "<<fits[8]->p1<<"     err p1 = "<<fits[8]->errp1<<endl;
    fit<<"p2 = "<<fits[8]->p2<<"     err p2 = "<<fits[8]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[8]->p0<<"  +-  "<<fits[8]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[8]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 5 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[17]->chi2)/(fits[17]->ndf)<<endl;
    fit<<"NdF        = "<<fits[17]->ndf<<endl;
    fit<<"p0 = "<<fits[17]->p0<<"     err p0 = "<<fits[17]->errp0<<endl;
    fit<<"p1 = "<<fits[17]->p1<<"     err p1 = "<<fits[17]->errp1<<endl;
    fit<<"p2 = "<<fits[17]->p2<<"     err p2 = "<<fits[17]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[17]->p0<<"  +-  "<<fits[17]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[17]->min<<endl<<endl;
    
    fit<<endl<<"--------------------------- SUMMARY FIT: TOB LAYER 6 -------------------------"<<endl<<endl;
    fit<<"Chi Square/ndf = "<<(fits[18]->chi2)/(fits[18]->ndf)<<endl;
    fit<<"NdF        = "<<fits[18]->ndf<<endl;
    fit<<"p0 = "<<fits[18]->p0<<"     err p0 = "<<fits[18]->errp0<<endl;
    fit<<"p1 = "<<fits[18]->p1<<"     err p1 = "<<fits[18]->errp1<<endl;
    fit<<"p2 = "<<fits[18]->p2<<"     err p2 = "<<fits[18]->errp2<<endl<<endl;
    fit<<"Minimum at angle = "<<fits[18]->p0<<"  +-  "<<fits[18]->errp0<<endl;
    fit<<"Cluster size at the minimum = "<<fits[18]->min<<endl<<endl;
  }

  if(conf_.getParameter<bool>("SINGLE_DETECTORS")) {
    for(Iditer=Detvector.begin(); Iditer!=Detvector.end(); ++Iditer){

      fit<<endl<<"-------------------------- MODULE HISTOGRAM FIT ------------------------"<<endl<<endl;
      fit<<makedescription(*Iditer)<<endl<<endl;
      fit<<"Chi Square/ndf = "<<(fits[Iditer->rawId()]->chi2)/(fits[Iditer->rawId()]->ndf)<<endl;
      fit<<"NdF        = "<<fits[Iditer->rawId()]->ndf<<endl;
      fit<<"p0 = "<<fits[Iditer->rawId()]->p0<<"     err p0 = "<<fits[Iditer->rawId()]->errp0<<endl;
      fit<<"p1 = "<<fits[Iditer->rawId()]->p1<<"     err p1 = "<<fits[Iditer->rawId()]->errp1<<endl;
      fit<<"p2 = "<<fits[Iditer->rawId()]->p2<<"     err p2 = "<<fits[Iditer->rawId()]->errp2<<endl<<endl;
      fit<<"Minimum at angle = "<<fits[Iditer->rawId()]->p0<<"  +-  "<<fits[Iditer->rawId()]->errp0<<endl;
      fit<<"Cluster size at the minimum = "<<fits[Iditer->rawId()]->min<<endl<<endl;
    }
  } // end if SINGLE_DETECTORS = true

  fit.close();

  //Set directories

  std::cout << "Setting directories" << std::endl;
  
  for(n=1;n<nmax;++n){
    histos[n]->SetDirectory(summary);}
  
  histos[19]->SetDirectory( summary);

  poClusterChargeTH1F->SetDirectory( summary); 
  hphi->SetDirectory(summary);
  hnhit->SetDirectory(summary);
  hEventTrackNumber->SetDirectory(summary);

  oGlobalPlots[0]->SetDirectory( summary);
  for( std::vector<TH1D *>::iterator oIter = oDetPlots[StripSubdetector::TIB].begin();
       oIter != oDetPlots[StripSubdetector::TIB].end();
       ++oIter) {
    ( *oIter)->SetDirectory( summary);
  }

  for( std::vector<TH1D *>::iterator oIter = oDetPlots[StripSubdetector::TOB].begin();
       oIter != oDetPlots[StripSubdetector::TOB].end();
       ++oIter) {
    ( *oIter)->SetDirectory( summary);
  }

  if(conf_.getParameter<bool>("TIB_ON")){
    htaTIBL1mono->SetDirectory(summary);
    htaTIBL1stereo->SetDirectory(summary);
    htaTIBL2mono->SetDirectory(summary);
    htaTIBL2stereo->SetDirectory(summary);
    htaTIBL3->SetDirectory(summary);
    htaTIBL4->SetDirectory(summary);
  }

  if(conf_.getParameter<bool>("TOB_ON")){
    htaTOBL1mono->SetDirectory(summary);
    htaTOBL1stereo->SetDirectory(summary);
    htaTOBL2mono->SetDirectory(summary);
    htaTOBL2stereo->SetDirectory(summary);
    htaTOBL3->SetDirectory(summary);
    htaTOBL4->SetDirectory(summary);
    htaTOBL5->SetDirectory(summary);
    htaTOBL6->SetDirectory(summary);
  }

  hwvst->SetDirectory(summary);  
  
  if(conf_.getParameter<bool>("SINGLE_DETECTORS")){

    std::cout << "starting loop on Detvector" << std::endl;
    for(Iditer=Detvector.begin(); Iditer!=Detvector.end(); ++Iditer){
  
      StripSubdetector DetId(Iditer->rawId());
  
      if(conf_.getParameter<bool>("TIB_ON")){
	if(Iditer->subdetId() == int (StripSubdetector::TIB)){
      
	  TIBDetId TIBid=TIBDetId(DetId);
    
	  int correctedlayer = TIBid.layer();
      
	  if(TIBid.string()[0] == 0){   
	    if(correctedlayer == 1){
	      histos[Iditer->rawId()]->SetDirectory(TIBbw1);}
	    if(correctedlayer == 2){
	      histos[Iditer->rawId()]->SetDirectory(TIBbw2);}
	    if(correctedlayer == 3){
	      histos[Iditer->rawId()]->SetDirectory(TIBbw3);}
	    if(correctedlayer == 4){
	      histos[Iditer->rawId()]->SetDirectory(TIBbw4);}
	  }
	  if(TIBid.string()[0] == 1){
	    if(correctedlayer == 1){
	      histos[Iditer->rawId()]->SetDirectory(TIBfw1);}
	    if(correctedlayer == 2){
	      histos[Iditer->rawId()]->SetDirectory(TIBfw2);}
	    if(correctedlayer == 3){
	      histos[Iditer->rawId()]->SetDirectory(TIBfw3);}
	    if(correctedlayer == 4){
	      histos[Iditer->rawId()]->SetDirectory(TIBfw4);}
	  }
	}
      }
      
      if(conf_.getParameter<bool>("TID_ON")) {
	if(Iditer->subdetId() == int (StripSubdetector::TID)){
  
	  TIDDetId TIDid=TIDDetId(DetId);
     
	  if(TIDid.module()[0] == 0){ 
	    if(TIDid.wheel() == 1){
	      histos[Iditer->rawId()]->SetDirectory(TIDbw1);}
	    if(TIDid.wheel() == 2){
	      histos[Iditer->rawId()]->SetDirectory(TIDbw2);}
	    if(TIDid.wheel() == 3){
	      histos[Iditer->rawId()]->SetDirectory(TIDbw3);}
	  } 
	  if(TIDid.module()[0] == 1){
	    if(TIDid.wheel() == 1){
	      histos[Iditer->rawId()]->SetDirectory(TIDfw1);}
	    if(TIDid.wheel() == 2){
	      histos[Iditer->rawId()]->SetDirectory(TIDfw2);}
	    if(TIDid.wheel() == 3){
	      histos[Iditer->rawId()]->SetDirectory(TIDfw3);}
	  }
	}
      }

      if(conf_.getParameter<bool>("TOB_ON")){
	if(Iditer->subdetId() == int (StripSubdetector::TOB)){
    
	  TOBDetId TOBid=TOBDetId(DetId);
    
	  int correctedlayer = TOBid.layer();
    
	  if(TOBid.rod()[0] == 0){      
	    if(correctedlayer == 1){
	      histos[Iditer->rawId()]->SetDirectory(TOBbw1);}
	    if(correctedlayer == 2){
	      histos[Iditer->rawId()]->SetDirectory(TOBbw2);}
	    if(correctedlayer == 3){
	      histos[Iditer->rawId()]->SetDirectory(TOBbw3);}
	    if(correctedlayer == 4){
	      histos[Iditer->rawId()]->SetDirectory(TOBbw4);}
	    if(correctedlayer == 5){
	      histos[Iditer->rawId()]->SetDirectory(TOBbw5);}
	    if(correctedlayer == 6){
	      histos[Iditer->rawId()]->SetDirectory(TOBbw6);}
	  }
	  if(TOBid.rod()[0] == 1){
	    if(correctedlayer == 1){
	      histos[Iditer->rawId()]->SetDirectory(TOBfw1);}
	    if(correctedlayer == 2){
	      histos[Iditer->rawId()]->SetDirectory(TOBfw2);}
	    if(correctedlayer == 3){
	      histos[Iditer->rawId()]->SetDirectory(TOBfw3);}
	    if(correctedlayer == 4){
	      histos[Iditer->rawId()]->SetDirectory(TOBfw4);}
	    if(correctedlayer == 5){
	      histos[Iditer->rawId()]->SetDirectory(TOBfw5);}
	    if(correctedlayer == 6){
	      histos[Iditer->rawId()]->SetDirectory(TOBfw6);}
	  }
	}
      }
     
      if(conf_.getParameter<bool>("TEC_ON")){
	if(Iditer->subdetId() == int (StripSubdetector::TEC)){
    
	  TECDetId TECid=TECDetId(DetId);
     
	  if(TECid.petal()[0] == 0){     
	    if(TECid.wheel() == 1){
	      histos[Iditer->rawId()]->SetDirectory(TECbw1);}
	    if(TECid.wheel() == 2){
	      histos[Iditer->rawId()]->SetDirectory(TECbw2);}
	    if(TECid.wheel() == 3){
	      histos[Iditer->rawId()]->SetDirectory(TECbw3);}
	    if(TECid.wheel() == 4){
	      histos[Iditer->rawId()]->SetDirectory(TECbw4);}
	    if(TECid.wheel() == 5){
	      histos[Iditer->rawId()]->SetDirectory(TECbw5);}
	    if(TECid.wheel() == 6){
	      histos[Iditer->rawId()]->SetDirectory(TECbw6);}
	    if(TECid.wheel() == 7){
	      histos[Iditer->rawId()]->SetDirectory(TECbw7);}
	    if(TECid.wheel() == 8){
	      histos[Iditer->rawId()]->SetDirectory(TECbw8);}
	    if(TECid.wheel() == 9){
	      histos[Iditer->rawId()]->SetDirectory(TECbw9);}
	  }
	  if(TECid.petal()[0] == 1){
	    if(TECid.wheel() == 1){
	      histos[Iditer->rawId()]->SetDirectory(TECfw1);}
	    if(TECid.wheel() == 2){
	      histos[Iditer->rawId()]->SetDirectory(TECfw2);}
	    if(TECid.wheel() == 3){
	      histos[Iditer->rawId()]->SetDirectory(TECfw3);}
	    if(TECid.wheel() == 4){
	      histos[Iditer->rawId()]->SetDirectory(TECfw4);}
	    if(TECid.wheel() == 5){
	      histos[Iditer->rawId()]->SetDirectory(TECfw5);}
	    if(TECid.wheel() == 6){
	      histos[Iditer->rawId()]->SetDirectory(TECfw6);}
	    if(TECid.wheel() == 7){
	      histos[Iditer->rawId()]->SetDirectory(TECfw7);}
	    if(TECid.wheel() == 8){
	      histos[Iditer->rawId()]->SetDirectory(TECfw8);}
	    if(TECid.wheel() == 9){
	      histos[Iditer->rawId()]->SetDirectory(TECfw9);}
	  }
	}  
      }
    }
  }
  hFile->Write();
  hFile->Close();
}

// Declare summary histograms
// --------------------------
void TIFNtupleMaker::_summaryHistos() {
  //Summary histograms

  TAxis *xaxis, *yaxis;
  
  if(conf_.getParameter<bool>("TIB_ON")){
    std::string TIB_name;
    stringstream Layer_strstream;
    for (int i=0; i<4; ++i) {
      Layer_strstream << (i+1);
      TIB_name = "TIBL" + Layer_strstream.str();
      std::cout << "TIB_name = " << TIB_name.c_str() << std::endl;
      //      htaTIBmono[n] = new TH1F(TIB_name.c_str(),"Track angle (TIB L1) MONO",120,-60.,60.);
      // empty the layer number
      Layer_strstream.str("");
    }
    // MODIFYING









    htaTIBL1mono = new TH1F("TIBL1angle_mono","Track angle (TIB L1) MONO",120,-60.,60.);
    xaxis = htaTIBL1mono->GetXaxis();
    xaxis->SetTitle("degree");
    htaTIBL1stereo = new TH1F("TIBL1angle_stereo","Track angle (TIB L1) STEREO",120,-60.,60.);
    xaxis = htaTIBL1stereo->GetXaxis();
    xaxis->SetTitle("degree");
    htaTIBL2mono = new TH1F("TIBL2angle_mono","Track angle (TIB L2) MONO",120,-60.,60.);
    xaxis = htaTIBL2mono->GetXaxis();
    xaxis->SetTitle("degree");
    htaTIBL2stereo = new TH1F("TIBL2angle_stereo","Track angle (TIB L2) STEREO",120,-60.,60.);
    xaxis = htaTIBL2stereo->GetXaxis();
    xaxis->SetTitle("degree");
    htaTIBL3 = new TH1F("TIBL3angle","Track angle (TIB L3)",120,-60.,60.);
    xaxis = htaTIBL3->GetXaxis();
    xaxis->SetTitle("degree");
    htaTIBL4 = new TH1F("TIBL4angle","Track angle (TIB L4)",120,-60.,60.);
    xaxis = htaTIBL4->GetXaxis();
    xaxis->SetTitle("degree");
  }

  // TOB can be turned off with the bool, default is ON
  if(conf_.getParameter<bool>("TOB_ON")){
    htaTOBL1mono = new TH1F("TOBL1angle_mono","Track angle (TOB L1) MONO",120,-60.,60.);
    xaxis = htaTOBL1mono->GetXaxis();
    xaxis->SetTitle("degree");
    htaTOBL1stereo = new TH1F("TOBL1angle_stereo","Track angle (TOB L1) STEREO",120,-60.,60.);
    xaxis = htaTOBL1stereo->GetXaxis();
    xaxis->SetTitle("degree");
    htaTOBL2mono = new TH1F("TOBL2angle_mono","Track angle (TOB L2) MONO",120,-60.,60.);
    xaxis = htaTOBL2mono->GetXaxis();
    xaxis->SetTitle("degree");
    htaTOBL2stereo = new TH1F("TOBL2angle_stereo","Track angle (TOB L2) STEREO",120,-60.,60.);
    xaxis = htaTOBL2stereo->GetXaxis();
    xaxis->SetTitle("degree");    
    htaTOBL3 = new TH1F("TOBL3","Track angle (TOB L3)",120,-60.,60.);
    xaxis = htaTOBL3->GetXaxis();
    xaxis->SetTitle("degree");
    htaTOBL4 = new TH1F("TOBL4","Track angle (TOB L4)",120,-60.,60.);
    xaxis = htaTOBL4->GetXaxis();
    xaxis->SetTitle("degree");
    htaTOBL5 = new TH1F("TOBL5","Track angle (TOB L5)",120,-60.,60.);
    xaxis = htaTOBL5->GetXaxis();
    xaxis->SetTitle("degree");
    htaTOBL6 = new TH1F("TOBL6","Track angle (TOB L6)",120,-60.,60.);
    xaxis = htaTOBL6->GetXaxis();
    xaxis->SetTitle("degree");
  }

  // TIB
  // ---
  //  if(conf_.getParameter<bool>("TIB_ON")){
  histos[1] = new TProfile("TIBL1_widthvsangle", "Cluster width vs track angle: TIB layer 1",30,-30.,30.);
  xaxis = histos[1]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[1]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[2] = new TProfile("TIBL1_widthvsangle_int", "Cluster width vs track angle: TIB layer 1 INT",30,-30.,30.);
  xaxis = histos[2]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[2]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[3] = new TProfile("TIBL1_widthvsangle_ext", "Cluster width vs track angle: TIB layer 1 EXT",30,-30.,30.);
  xaxis = histos[3]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[3]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[4] = new TProfile("TIBL2_widthvsangle", "Cluster width vs track angle: TIB layer 2",30,-30.,30.);
  xaxis = histos[4]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[4]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[5] = new TProfile("TIBL2_widthvsangle_int", "Cluster width vs track angle: TIB layer 2 INT",30,-30.,30.);
  xaxis = histos[5]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[5]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[6] = new TProfile("TIBL2_widthvsangle_ext", "Cluster width vs track angle: TIB layer 2 EXT",30,-30.,30.);
  xaxis = histos[6]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[6]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[7] = new TProfile("TIBL3_widthvsangle", "Cluster width vs track angle: TIB layer 3",30,-30.,30.);
  xaxis = histos[7]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[7]->GetYaxis();
  yaxis->SetTitle("number of strips");  
  histos[8] = new TProfile("TIBL3_widthvsangle_int", "Cluster width vs track angle: TIB layer 3 INT",30,-30.,30.);
  xaxis = histos[8]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[8]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[9] = new TProfile("TIBL3_widthvsangle_ext", "Cluster width vs track angle: TIB layer 3 EXT",30,-30.,30.);
  xaxis = histos[9]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[9]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[10] = new TProfile("TIBL4_widthvsangle", "Cluster width vs track angle: TIB layer 4",30,-30.,30.);
  xaxis = histos[10]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[10]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[11] = new TProfile("TIBL4_widthvsangle_int", "Cluster width vs track angle: TIB layer 4 INT",30,-30.,30.);
  xaxis = histos[11]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[11]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[12] = new TProfile("TIBL4_widthvsangle_ext", "Cluster width vs track angle: TIB layer 4 EXT",30,-30.,30.);
  xaxis = histos[12]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[12]->GetYaxis();
  yaxis->SetTitle("number of strips");
  //  } // TIB_ON

  // TOB
  // ---
  //  if(conf_.getParameter<bool>("TOB_ON")) {
  histos[13] = new TProfile("TOBL1_widthvsangle", "Cluster width vs track angle: TOB layer 1",30,-30.,30.);
  xaxis = histos[13]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[13]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[14] = new TProfile("TOBL2_widthvsangle", "Cluster width vs track angle: TOB layer 2",30,-30.,30.);
  xaxis = histos[14]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[14]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[15] = new TProfile("TOBL3_widthvsangle", "Cluster width vs track angle: TOB layer 3",30,-30.,30.);
  xaxis = histos[15]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[15]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[16] = new TProfile("TOBL4_widthvsangle", "Cluster width vs track angle: TOB layer 4",30,-30.,30.);
  xaxis = histos[16]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[16]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[17] = new TProfile("TOBL5_widthvsangle", "Cluster width vs track angle: TOB layer 5",30,-30.,30.);
  xaxis = histos[17]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[17]->GetYaxis();
  yaxis->SetTitle("number of strips");
  histos[18] = new TProfile("TOBL6_widthvsangle", "Cluster width vs track angle: TOB layer 6",30,-30.,30.);
  xaxis = histos[18]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[18]->GetYaxis();
  yaxis->SetTitle("number of strips");


  // -------------------------
  // MAKE IT GENERAL OR REMOVE
  // -------------------------

  // TOB Layer 6 cluster charge vs angle
  // -----------------------------------
  histos[19] = new TProfile("Charge_vs_Angle", "Cluster charge vs track angle: TOB layer 6",30,-30.,30.);
  xaxis = histos[19]->GetXaxis();
  xaxis->SetTitle("degree");
  yaxis = histos[19]->GetYaxis();
  yaxis->SetTitle("charge");

  // -------------------------
  // -------------------------

  //  } // if TOB_ON = true

  // Cluster charge
  // --------------
  poClusterChargeTH1F = new TH1F( "Cluster_Charge", "Cluster Charge",600,0.,300.);
  xaxis = poClusterChargeTH1F->GetXaxis();
  xaxis->SetTitle("Charge");

  TH1D *poNewPlot;
  // Global
  poNewPlot = new TH1D( "NumberOfTrackClusterDivTotNumOfClusters", "Number of Track Clusters / Total number of Clusters",150,0.,1.5);
  poNewPlot->GetXaxis()->SetTitle( "TrackClusters / AllClusters");
  oGlobalPlots.push_back( poNewPlot);

  // TIB
  if(conf_.getParameter<bool>("TIB_ON")){
    poNewPlot = new TH1D( "NumberOfClustersInTIBL1", "TIB L1: Number of clusters per layer",10,0.,20.);
    poNewPlot->GetXaxis()->SetTitle( "Clusters");
    oDetPlots[StripSubdetector::TIB].push_back( poNewPlot);

    poNewPlot = new TH1D( "NumberOfClustersInTIBL2", "TIB L2: Number of clusters per layer",10,0.,20.);
    poNewPlot->GetXaxis()->SetTitle( "Clusters");
    oDetPlots[StripSubdetector::TIB].push_back( poNewPlot);

    poNewPlot = new TH1D( "NumberOfClustersInTIBL3", "TIB L3: Number of clusters per layer",10,0.,20.);
    poNewPlot->GetXaxis()->SetTitle( "Clusters");
    oDetPlots[StripSubdetector::TIB].push_back( poNewPlot);

    poNewPlot = new TH1D( "NumberOfClustersInTIBL4", "TIB L4: Number of clusters per layer",10,0.,20.);
    poNewPlot->GetXaxis()->SetTitle( "Clusters");
    oDetPlots[StripSubdetector::TIB].push_back( poNewPlot);
  }

  // TOB
  if(conf_.getParameter<bool>("TOB_ON")) {
    poNewPlot = new TH1D( "NumberOfClustersInTOBL3", "TOB L3: Number of clusters per layer",10,0.,10.);
    poNewPlot->GetXaxis()->SetTitle( "Clusters");
    oDetPlots[StripSubdetector::TOB].push_back( poNewPlot);

    poNewPlot = new TH1D( "NumberOfClustersInTOBL4", "TOB L4: Number of clusters per layer",10,0.,10.);
    poNewPlot->GetXaxis()->SetTitle( "Clusters");
    oDetPlots[StripSubdetector::TOB].push_back( poNewPlot);
  }

  monodscounter=0;
  monosscounter=0;
  stereocounter=0;

  if(conf_.getParameter<bool>("SINGLE_DETECTORS")){

    //Get Ids;
  
    // Modified from estracker to tracker
    const TrackerGeometry::DetIdContainer& Id = tracker->detIds();
    TrackerGeometry::DetIdContainer::const_iterator Iditer;

    for(Iditer=Id.begin();Iditer!=Id.end();++Iditer){
      if((Iditer->subdetId() != uint32_t(PixelSubdetector::PixelBarrel)) && (Iditer->subdetId() != uint32_t(PixelSubdetector::PixelEndcap))){
	StripSubdetector subid(*Iditer);

	//Mono single sided detectors

	// TIB
	// ---
	if(conf_.getParameter<bool>("TIB_ON")){
	  if(subid.glued() == 0){
	    if(subid.subdetId() == int (StripSubdetector::TIB)){
	      TIBDetId TIBid=TIBDetId(*Iditer);
	      if((TIBid.layer()!=1) && (TIBid.layer()!=2)){			
		++monosscounter;
		Detvector.push_back(*Iditer);
		histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-30.,30.);
		xaxis = histos[Iditer->rawId()]->GetXaxis();
		xaxis->SetTitle("degree");
		yaxis = histos[Iditer->rawId()]->GetYaxis();
		yaxis->SetTitle("number of strips");
	      }
	    }
	  }
	}
	// TOB
	// ---
	if(conf_.getParameter<bool>("TOB_ON")){
	  if(subid.subdetId() == int (StripSubdetector::TOB)){
	    TOBDetId TOBid=TOBDetId(*Iditer);
	    if((TOBid.layer()!=1) && (TOBid.layer()!=2)){
	      ++monosscounter;
	      Detvector.push_back(*Iditer);
	      histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-30.,30.);
	      xaxis = histos[Iditer->rawId()]->GetXaxis();
	      xaxis->SetTitle("degree");
	      yaxis = histos[Iditer->rawId()]->GetYaxis();
	      yaxis->SetTitle("number of strips");
	    }
	  }
	}
	// TID
	// ---
	if(conf_.getParameter<bool>("TID_ON")){
	  if(subid.subdetId() == int (StripSubdetector::TID)){
	    TIDDetId TIDid=TIDDetId(*Iditer);
	    if((TIDid.ring()!=1) && (TIDid.ring()!=2)){
	      ++monosscounter;
	      Detvector.push_back(*Iditer);
	      histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-30.,30.);
	      xaxis = histos[Iditer->rawId()]->GetXaxis();
	      xaxis->SetTitle("degree");
	      yaxis = histos[Iditer->rawId()]->GetYaxis();
	      yaxis->SetTitle("number of strips");
	    }
	  }
	}
	// TEC
	// ---
	if(conf_.getParameter<bool>("TEC_ON")){
	  if(subid.subdetId() == int (StripSubdetector::TEC)){
	    TECDetId TECid=TECDetId(*Iditer);
	    if((TECid.ring()!=1) && (TECid.ring()!=2) && (TECid.ring()!=5)){
	      ++monosscounter;
	      Detvector.push_back(*Iditer);
	      histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-30.,30.);
	      xaxis = histos[Iditer->rawId()]->GetXaxis();
	      xaxis->SetTitle("degree");
	      yaxis = histos[Iditer->rawId()]->GetYaxis();
	      yaxis->SetTitle("number of strips");
	    }
	  }
	}
      
	//Mono double sided detectors

	if((subid.glued() != 0) && (subid.stereo() == 0)){
	  ++monodscounter;
	  Detvector.push_back(*Iditer);
	  histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-30.,30.);
	  xaxis = histos[Iditer->rawId()]->GetXaxis();
	  xaxis->SetTitle("degree");
	  yaxis = histos[Iditer->rawId()]->GetYaxis();
	  yaxis->SetTitle("number of strips");
	}
      
	//Stereo detectors
      
	if((subid.glued() != 0) && (subid.stereo() == 1)){
	  ++stereocounter;
	  Detvector.push_back(*Iditer);
	  histos[Iditer->rawId()] = new TProfile(makename(*Iditer),makedescription(*Iditer),30,-30.,30.);
	  xaxis = histos[Iditer->rawId()]->GetXaxis();
	  xaxis->SetTitle("degree");
	  yaxis = histos[Iditer->rawId()]->GetYaxis();
	  yaxis->SetTitle("number of strips");
	} 
      }
    }
  } // end if SINGLE_DETECTORS = true
}

// Create directory hierarchy  
// --------------------------
void TIFNtupleMaker::_directoryHierarchy() {
  histograms = new TDirectory("Histograms", "Histograms", "");
  summary = new TDirectory("Summary", "Summary", "");  
  
  //TIB-TID-TOB-TEC    
  if(conf_.getParameter<bool>("TIB_ON")){
    // plus Forward-Backward
    TIB = histograms->mkdir("TIB");
    TIBfw = TIB->mkdir("TIB forward");
    TIBbw = TIB->mkdir("TIB backward");

    //TIB directories
  
    TIBfw1 = TIBfw->mkdir("TIB forward layer 1");
    TIBfw2 = TIBfw->mkdir("TIB forward layer 2");
    TIBfw3 = TIBfw->mkdir("TIB forward layer 3");
    TIBfw4 = TIBfw->mkdir("TIB forward layer 4");
  
    TIBbw1 = TIBbw->mkdir("TIB backward layer 1");
    TIBbw2 = TIBbw->mkdir("TIB backward layer 2");
    TIBbw3 = TIBbw->mkdir("TIB backward layer 3");
    TIBbw4 = TIBbw->mkdir("TIB backward layer 4");
  }

  if(conf_.getParameter<bool>("TOB_ON")) {
    TOB = histograms->mkdir("TOB");
    TOBfw = TOB->mkdir("TOB forward");
    TOBbw = TOB->mkdir("TOB backward");

    //TOB directories
  
    TOBfw1 = TOBfw->mkdir("TOB forward layer 1");
    TOBfw2 = TOBfw->mkdir("TOB forward layer 2");
    TOBfw3 = TOBfw->mkdir("TOB forward layer 3");
    TOBfw4 = TOBfw->mkdir("TOB forward layer 4");
    TOBfw5 = TOBfw->mkdir("TOB forward layer 5");
    TOBfw6 = TOBfw->mkdir("TOB forward layer 6");
  
    TOBbw1 = TOBbw->mkdir("TOB backward layer 1");
    TOBbw2 = TOBbw->mkdir("TOB backward layer 2");
    TOBbw3 = TOBbw->mkdir("TOB backward layer 3");
    TOBbw4 = TOBbw->mkdir("TOB backward layer 4");
    TOBbw5 = TOBbw->mkdir("TOB backward layer 5");
    TOBbw6 = TOBbw->mkdir("TOB backward layer 6");
  }

  if(conf_.getParameter<bool>("TID_ON")) {
    TID = histograms->mkdir("TID");
    TIDfw = TID->mkdir("TID forward");
    TIDbw = TID->mkdir("TID backward");

    //TID directories

    TIDfw1 = TIDfw->mkdir("TID forward wheel 1");
    TIDfw2 = TIDfw->mkdir("TID forward wheel 2");
    TIDfw3 = TIDfw->mkdir("TID forward wheel 3");
  
    TIDbw1 = TIDbw->mkdir("TID backward wheel 1");
    TIDbw2 = TIDbw->mkdir("TID backward wheel 2");
    TIDbw3 = TIDbw->mkdir("TID backward wheel 3"); 
  }

  if(conf_.getParameter<bool>("TEC_ON")) {
    TEC = histograms->mkdir("TEC");
    TECfw = TEC->mkdir("TEC forward");
    TECbw = TEC->mkdir("TEC backward"); 

    //TEC directories
  
    TECfw1 = TECfw->mkdir("TEC forward wheel 1");
    TECfw2 = TECfw->mkdir("TEC forward wheel 2");
    TECfw3 = TECfw->mkdir("TEC forward wheel 3");
    TECfw4 = TECfw->mkdir("TEC forward wheel 4");
    TECfw5 = TECfw->mkdir("TEC forward wheel 5");
    TECfw6 = TECfw->mkdir("TEC forward wheel 6");
    TECfw7 = TECfw->mkdir("TEC forward wheel 7");
    TECfw8 = TECfw->mkdir("TEC forward wheel 8");
    TECfw9 = TECfw->mkdir("TEC forward wheel 9");
  
    TECbw1 = TECbw->mkdir("TEC backward layer 1");
    TECbw2 = TECbw->mkdir("TEC backward layer 2");
    TECbw3 = TECbw->mkdir("TEC backward layer 3");
    TECbw4 = TECbw->mkdir("TEC backward layer 4");
    TECbw5 = TECbw->mkdir("TEC backward layer 5");
    TECbw6 = TECbw->mkdir("TEC backward layer 6");
    TECbw7 = TECbw->mkdir("TEC backward layer 7");
    TECbw8 = TECbw->mkdir("TEC backward layer 8");
    TECbw9 = TECbw->mkdir("TEC backward layer 9");
  }
}

//Makename function

const char* TIFNtupleMaker::makename(DetId detid){
  
  std::string name;
  
  stringstream idnum;
  stringstream layernum;
  stringstream wheelnum;
  stringstream stringnum;
  stringstream rodnum;
  stringstream ringnum;
  stringstream petalnum;
  
  idnum << detid.rawId();
  
  StripSubdetector DetId(detid.rawId());
  
  //TIB
  if(conf_.getParameter<bool>("TOB_ON")){  
    if(detid.subdetId() == int (StripSubdetector::TIB)){
      name="TIB";
    
      TIBDetId TIBid=TIBDetId(DetId);
    
      if(TIBid.string()[0] == 0){
	name+="bw";}
      if(TIBid.string()[0] == 1){
	name+="fw";}
    
      name+="L";
      int layer = TIBid.layer();    
      layernum << layer;
      name+=layernum.str();
    
      if(TIBid.string()[1] == 0){
	name+="int";}
      if(TIBid.string()[1] == 1){
	name+="ext";}
    
      name+="string";
      int string = TIBid.string()[2];
      stringnum << string;
      name+=stringnum.str();
    
      if(TIBid.stereo() == 0){
	name+="mono";}
      if(TIBid.stereo() == 1){
	name+="stereo";}    
    }
  }
  
  //TID
  if(conf_.getParameter<bool>("TID_ON")){
    if(detid.subdetId() == int (StripSubdetector::TID)){
      name="TID";
    
      TIDDetId TIDid=TIDDetId(DetId);
    
      if(TIDid.module()[0] == 0){
	name+="bw";}
      if(TIDid.module()[0] == 1){
	name+="fw";}
      
      name+="W";
      int wheel = TIDid.wheel();    
      wheelnum << wheel;
      name+=wheelnum.str();
    
      if(TIDid.side() == 1){
	name+="neg";}
      if(TIDid.side() == 2){
	name+="pos";}
      
      name+="ring";
      int ring = TIDid.ring();
      ringnum << ring;
      name+=ringnum.str();
    
      if(TIDid.stereo() == 0){
	name+="mono";}
      if(TIDid.stereo() == 1){
	name+="stereo";}    
    }
  }

  //TOB
  if(conf_.getParameter<bool>("TOB_ON")){
    if(detid.subdetId() == int (StripSubdetector::TOB)){
      name="TOB";
    
      TOBDetId TOBid=TOBDetId(DetId);
    
      if(TOBid.rod()[0] == 0){
	name+="bw";}
      if(TOBid.rod()[0] == 1){
	name+="fw";}
    
      name+="L";
      int layer = TOBid.layer(); 
      layernum << layer;
      name+=layernum.str();
 
      name+="rod";
      int rod = TOBid.rod()[1];
      rodnum << rod;
      name+=rodnum.str();
    
      if(TOBid.stereo() == 0){
	name+="mono";}
      if(TOBid.stereo() == 1){
	name+="stereo";}    
    }
  }

  //TEC
  if(conf_.getParameter<bool>("TEC_ON")){
    if(detid.subdetId() == int (StripSubdetector::TEC)){
      name="TEC";
    
      TECDetId TECid=TECDetId(DetId);
    
      if(TECid.petal()[0] == 0){
	name+="bw";}
      if(TECid.petal()[0] == 1){
	name+="fw";}
      
      name+="W";
      int wheel = TECid.wheel();    
      wheelnum << wheel;
      name+=wheelnum.str();
    
      if(TECid.side() == 1){
	name+="neg";}
      if(TECid.side() == 2){
	name+="pos";}
      
      name+="ring";
      int ring = TECid.ring();
      ringnum << ring;
      name+=ringnum.str();
    
      name+="petal";
      int petal = TECid.petal()[1];
      petalnum << petal;
      name+=petalnum.str();
    
      if(TECid.stereo() == 0){
	name+="mono";}
      if(TECid.stereo() == 1){
	name+="stereo";}    
    }
  }
    
  name+="_";
  
  name+=idnum.str();
  
  return name.c_str();
  
}

//Makedescription function

const char* TIFNtupleMaker::makedescription(DetId detid){
  
  std::string name;
  
  name="Cluster width vs track angle (";
  
  stringstream idnum;
  stringstream layernum;
  stringstream wheelnum;
  stringstream stringnum;
  stringstream rodnum;
  stringstream ringnum;
  stringstream petalnum;
  
  idnum << detid.rawId();
  
  StripSubdetector DetId(detid.rawId());
  
  //TIB
  if(conf_.getParameter<bool>("TOB_ON")){
    if(detid.subdetId() == int (StripSubdetector::TIB)){
      name+="TIB ";
    
      TIBDetId TIBid=TIBDetId(DetId);
    
      if(TIBid.string()[0] == 0){
	name+="backward, ";}
      if(TIBid.string()[0] == 1){
	name+="forward, ";}
    
      name+="Layer n.";
      int layer = TIBid.layer();  
      layernum << layer;
      name+=layernum.str();
    
      if(TIBid.string()[1] == 0){
	name+=", internal ";}
      if(TIBid.string()[1] == 1){
	name+=", external ";}
    
      name+="string n.";
      int string = TIBid.string()[2];
      stringnum << string;
      name+=stringnum.str();
    
      if(TIBid.stereo() == 0){
	name+=", mono,";}
      if(TIBid.stereo() == 1){
	name+=", stereo,";}  
    }
  }
  
  //TID
  if(conf_.getParameter<bool>("TID_ON")){
    if(detid.subdetId() == int (StripSubdetector::TID)){
      name+="TID ";
    
      TIDDetId TIDid=TIDDetId(DetId);
    
      if(TIDid.module()[0] == 0){
	name+="bacward, ";}
      if(TIDid.module()[0] == 1){
	name+="forward, ";}
      
      name+="Wheel n.";
      int wheel = TIDid.wheel();    
      wheelnum << wheel;
      name+=wheelnum.str();
    
      if(TIDid.side() == 1){
	name+=", negative ";}
      if(TIDid.side() == 2){
	name+=", positive ";}
      
      name+="ring n.";
      int ring = TIDid.ring();
      ringnum << ring;
      name+=ringnum.str();
    
      if(TIDid.stereo() == 0){
	name+=", mono";}
      if(TIDid.stereo() == 1){
	name+=", stereo";}    
    }
  }

  //TOB
  if(conf_.getParameter<bool>("TOB_ON")){
    if(detid.subdetId() == int (StripSubdetector::TOB)){
      name+="TOB ";
    
      TOBDetId TOBid=TOBDetId(DetId);
    
      if(TOBid.rod()[0] == 0){
	name+="backward, ";}
      if(TOBid.rod()[0] == 1){
	name+="forward, ";}
    
      name+="Layer n.";
      int layer = TOBid.layer(); 
      layernum << layer;
      name+=layernum.str();
    
      name+=", rod n.";
      int rod = TOBid.rod()[1];
      rodnum << rod;
      name+=rodnum.str();
    
      if(TOBid.stereo() == 0){
	name+=", mono,";}
      if(TOBid.stereo() == 1){
	name+=", stereo,";}    
    }
  }
  
  //TEC
  if(conf_.getParameter<bool>("TEC_ON")){
    if(detid.subdetId() == int (StripSubdetector::TEC)){
      name+="TEC ";
    
      TECDetId TECid=TECDetId(DetId);
    
      if(TECid.petal()[0] == 0){
	name+="backward, ";}
      if(TECid.petal()[0] == 1){
	name+="forward, ";}
      
      name+="Wheel n.";
      int wheel = TECid.wheel();    
      wheelnum << wheel;
      name+=wheelnum.str();
    
      if(TECid.side() == 1){
	name+=", negative ";}
      if(TECid.side() == 2){
	name+=", positive ";}

      name+="ring n.";
      int ring = TECid.ring();
      ringnum << ring;
      name+=ringnum.str();
    
      name+=", petal n.";
      int petal = TECid.petal()[1];
      petalnum << petal;
      name+=petalnum.str();
    
      if(TECid.stereo() == 0){
	name+=", mono";}
      if(TECid.stereo() == 1){
	name+=", stereo";}  
    }
  }
      
  name+=" IdNumber = ";
  
  name+=idnum.str();
  
  name+=")";
  
  return name.c_str();
}

// ClusterEta = SignalL / ( SignalL + SignalR)
// where:
//   SignalL and SignalR are two strips with the highest amplitudes. 
//   SignalL - is the strip with the smaller strip number
//   SignalR - with the highest strip number accordingly
// @arguments
//   roSTRIP_AMPLITUDES	 vector of strips ADC counts in cluster
//   rnFIRST_STRIP	 cluster first strip shift whithin module
//   roDIGIS		 vector of digis within current module
// @return
//   int  ClusterEta or -99 on error
double TIFNtupleMaker::getClusterEta( const std::vector<uint16_t> &roSTRIP_AMPLITUDES,
				      const int			  &rnFIRST_STRIP,
				      const DigisVector		  &roDIGIS) const {
  // Given value is used to separate non-physical values
  // [Example: cluster with empty amplitudes vector]
  double dClusterEta = -99;

  // Cluster eta calculation
  int anMaxSignal[2][2];

  // Null array before using it
  for( int i = 0; 2 > i; ++i) {
    for( int j = 0; 2 > j; ++j) {
      anMaxSignal[i][j] = 0;
    }
  }
	
  // Find two strips with highest amplitudes
  // i is a stip number within module
  for( int i = 0, nSize = roSTRIP_AMPLITUDES.size(); nSize > i; ++i) {
    int nCurCharge = roSTRIP_AMPLITUDES[i];

    if( nCurCharge > anMaxSignal[1][1]) {
      anMaxSignal[0][0] = anMaxSignal[1][0];
      anMaxSignal[0][1] = anMaxSignal[1][1];
      // Convert to global strip number within module
      anMaxSignal[1][0] = i + rnFIRST_STRIP; 
      anMaxSignal[1][1] = nCurCharge;
    } else if( nCurCharge > anMaxSignal[0][1]) {
      // Convert to global strip number within module
      anMaxSignal[0][0] = i + rnFIRST_STRIP;
      anMaxSignal[0][1] = nCurCharge;
    }
  }
  
  if( ( anMaxSignal[1][1] + anMaxSignal[0][1]) != 0) {
    if( anMaxSignal[0][0] > anMaxSignal[1][0]) {
      // anMaxSignal[1] is Left one
      dClusterEta = ( 1.0 * anMaxSignal[1][1]) / ( anMaxSignal[1][1] + 
						   anMaxSignal[0][1]);
    } else if( 0 == anMaxSignal[0][0] && 
	       0 == anMaxSignal[0][1]) {

      // One Strip cluster: check for Digis
      DigisVector::const_iterator oITER( roDIGIS.begin());
      for( ;
	   oITER != roDIGIS.end() && oITER->strip() != anMaxSignal[1][0];
	   ++oITER) {}

      // Check if Digi for given cluster strip was found
      if( oITER != roDIGIS.end()) {

	// Check if previous neighbouring strip exists
	DigisVector::const_iterator oITER_PREV( roDIGIS.end());
	if( oITER != roDIGIS.begin() &&
	    ( oITER->strip() - 1) == ( oITER - 1)->strip()) {
	  // There is previous strip specified :)
	  oITER_PREV = oITER - 1;
	}

	// Check if next neighbouring strip exists
	DigisVector::const_iterator oITER_NEXT( roDIGIS.end());
	if( oITER != roDIGIS.end() &&
	    oITER != ( roDIGIS.end() - 1) &&
	    ( oITER->strip() + 1) == ( oITER + 1)->strip()) {
	  // There is previous strip specified :)
	  oITER_NEXT = oITER + 1;
	}

	if( oITER_PREV != oITER_NEXT) {
	  if( oITER_PREV != roDIGIS.end() && oITER_NEXT != roDIGIS.end()) {
	    // Both Digis are specified
	    // Now Pick the one with max amplitude
	    if( oITER_PREV->adc() > oITER_NEXT->adc()) {
	      dClusterEta = ( 1.0 * oITER_PREV->adc()) / ( oITER_PREV->adc() + 
							   anMaxSignal[1][1]);
	    } else {
	      dClusterEta = ( 1.0 * anMaxSignal[1][1]) / ( oITER_NEXT->adc() + 
							   anMaxSignal[1][1]);
	    }
	  } else if( oITER_PREV != roDIGIS.end()) {
	    // only Prev digi is specified
	    dClusterEta = ( 1.0 * oITER_PREV->adc()) / ( oITER_PREV->adc() + 
							 anMaxSignal[1][1]);
	  } else {
	    // only Next digi is specified
	    dClusterEta = ( 1.0 * anMaxSignal[1][1]) / ( oITER_NEXT->adc() + 
							 anMaxSignal[1][1]);
	  }
	} else {
	  // PREV and NEXT iterators point to the end of DIGIs vector. 
	  // Consequently it is assumed there are no neighbouring digis at all
	  // for given cluster. It is obvious why ClusterEta should be Zero.
	  // [Hint: take a look at the case [0][0] < [1][0] ]
	  dClusterEta = 0;
	} // End check if any neighbouring digi is specified
      } else {
	// Digi for given Clusters strip was not found
	dClusterEta = 0;
      } // end check if Digi for given cluster strip was found
    } else {
      // anMaxSignal[0] is Left one
      dClusterEta = ( 1.0 * anMaxSignal[0][1]) / ( anMaxSignal[1][1] + 
						   anMaxSignal[0][1]);
    }
  } 
  return dClusterEta;
}

// ClusterCrosstalk = ( SignalL + SignalR) / SignalSeed
//   Extremely useful for croostalk determination that should be used in
// MonteCarlo clusters simulation. In MonteCarlo each cluster affects 
// neighbouring ones, the charge is divided:
//
//  Simulated             Digitized
//    Signal                Signal
//       +
//       +
//       +                    +
//       +                    +
//       +          =>        +
//       +                    +
//       +                    +
//       +                +   +   +
//       +                +   +   +
//  N-1  N  N+1          N-1  N  N+1
//
//  Strip   Crosstalk
//   N-1     x * SignalN
//   N       ( 1 - 2 * x) * SignalN
//   N+1     x * SignalN
//
// @arguments
//   roSTRIP_AMPLITUDES	 vector of strips ADC counts in cluster
//   rnFIRST_STRIP	 cluster first strip shift whithin module
//   roDIGIS		 vector of digis within current module
// @return
//   int  ClusterCrosstalk or -99 on error
double TIFNtupleMaker::getClusterCrossTalk( const std::vector<uint16_t> 
					    &roSTRIP_AMPLITUDES,
					    const int         &rnFIRST_STRIP,
					    const DigisVector &roDIGIS) const {
  // Given value is used to separate non-physical values
  // [Example: cluster with empty amplitudes vector]
  double dClusterCrossTalk = -99;

  switch( roSTRIP_AMPLITUDES.size()) {
  case 1: {
    // One Strip cluster: try to find corresponding Digi
    DigisVector::const_iterator oITER( roDIGIS.begin());
    for( ;
	 oITER != roDIGIS.end() && oITER->strip() != roSTRIP_AMPLITUDES[0];
	 ++oITER) {}

    // Check if Digi for given cluster strip was found
    if( oITER != roDIGIS.end()) {

      // Check if previous neighbouring strip exists
      DigisVector::const_iterator oITER_PREV( roDIGIS.end());
      if( oITER != roDIGIS.begin() &&
	  ( oITER->strip() - 1) == ( oITER - 1)->strip()) {
	// There is previous strip specified :)
	oITER_PREV = oITER - 1;
      }

      // Check if next neighbouring strip exists
      DigisVector::const_iterator oITER_NEXT( roDIGIS.end());
      if( oITER != roDIGIS.end() &&
	  oITER != ( roDIGIS.end() - 1) &&
	  ( oITER->strip() + 1) == ( oITER + 1)->strip()) {
	// There is previous strip specified :)
	oITER_NEXT = oITER + 1;
      }

      // Now check if both neighbouring digis exist
      if( oITER_PREV != roDIGIS.end() && oITER_NEXT != roDIGIS.end()) {
	// Both Digis are specified
	// Now Pick the one with max amplitude
	dClusterCrossTalk = 
	  calculateClusterCrossTalk( oITER_PREV->adc(),
				     roSTRIP_AMPLITUDES[0],
				     oITER_NEXT->adc());
      } // End check if both neighbouring digis exist
    } // end check if Digi for given cluster strip was found
  }
  case 3: {
    dClusterCrossTalk = calculateClusterCrossTalk( roSTRIP_AMPLITUDES[0],
						   roSTRIP_AMPLITUDES[1],
						   roSTRIP_AMPLITUDES[2]);
  }
  default:
    break;
  }
  return dClusterCrossTalk;
}

// Calculate Cluster CrossTalk:
// ClusterCrosstalk = ( SignalL + SignalR) / SignalSeed
// @arguments
//   rdADC_STRIPL  ADC in left strip
//   rnADC_STRIP   ADC in Center strip
//   rnADC_STRIPR  ADC in right strip
// @return
//   double  Calculated crosstalk or -99 on error 
double 
TIFNtupleMaker::calculateClusterCrossTalk( const double &rdADC_STRIPL,
					   const int    &rnADC_STRIP,
					   const int    &rnADC_STRIPR) const 
{
  // Check if neigbouring strips have signals amplitudes within some
  // error
  return ( abs( rdADC_STRIPL - rnADC_STRIPR) < 
	   dCROSS_TALK_ERR * ( rdADC_STRIPL + rnADC_STRIPR) / 2 &&
	   0 < rnADC_STRIP ? 
	   ( rdADC_STRIPL + rnADC_STRIPR) / rnADC_STRIP : 
	   -99);
}

/*
  double 
  TIFNtupleMaker::getClusterCrossTalk( const std::vector<uint16_t> 
  &roSTRIP_AMPLITUDES,
  const int         &rnFIRST_STRIP,
  const DigisVector &roDIGIS) const {
  // Given value is used to separate non-physical values
  // [Example: cluster with empty amplitudes vector]
  double dClusterCrossTalk = -99;

  switch( roSTRIP_AMPLITUDES.size()) {
  case 1: {
  // One Strip cluster: try to find corresponding Digi
  DigisVector::const_iterator oITER( roDIGIS.begin());
  for( ;
  oITER != roDIGIS.end() && oITER->strip() != roSTRIP_AMPLITUDES[0];
  ++oITER) {}

  // Check if Digi for given cluster strip was found
  if( oITER != roDIGIS.end()) {

  // Check if previous neighbouring strip exists
  DigisVector::const_iterator oITER_PREV( roDIGIS.end());
  if( oITER != roDIGIS.begin() &&
  ( oITER->strip() - 1) == ( oITER - 1)->strip()) {
  // There is previous strip specified :)
  oITER_PREV = oITER - 1;
  }

  // Check if next neighbouring strip exists
  DigisVector::const_iterator oITER_NEXT( roDIGIS.end());
  if( oITER != roDIGIS.end() &&
  oITER != ( roDIGIS.end() - 1) &&
  ( oITER->strip() + 1) == ( oITER + 1)->strip()) {
  // There is previous strip specified :)
  oITER_NEXT = oITER + 1;
  }

  // Now check if both neighbouring digis exist and there is no
  // anything in N-2 and N+2 to make sure neighbouring digis were
  // created from central one
  if( oITER_PREV != roDIGIS.end() && 
  oITER_NEXT != roDIGIS.end() &&
  !( oITER_PREV != roDIGIS.begin() && 
  ( oITER_PREV - 1)->strip() == oITER_PREV->strip() - 1) && 
  !( oITER_NEXT != roDIGIS.end() - 1 &&
  ( oITER_NEXT + 1)->strip() == oITER_NEXT->strip() + 1)) {

  // Both Digis are specified
  // Now Pick the one with max amplitude
  dClusterCrossTalk = 
  calculateClusterCrossTalk( oITER_PREV->adc(),
  roSTRIP_AMPLITUDES[0],
  oITER_NEXT->adc());
  } // End check if both neighbouring digis exist
  } // end check if Digi for given cluster strip was found
  }
  case 3: {
  // Try to find Digi that corresponds to central strip
  DigisVector::const_iterator oITER( roDIGIS.begin());
  for( ;
  oITER != roDIGIS.end() && oITER->strip() != roSTRIP_AMPLITUDES[1];
  ++oITER) {}

  // Check if Digi for given cluster strip was found
  if( oITER != roDIGIS.end()) {
  DigisVector::const_iterator oITER_N_MINUS_2( oITER);
  for( int i = 2; i > 0; --i) {
  if( oITER_N_MINUS_2 == roDIGIS.begin()) {
  // There is no N-2 Digi
  oITER_N_MINUS_2 = roDIGIS.end();
  break;
  }
  --oITER_N_MINUS_2;
  }

  DigisVector::const_iterator oITER_N_PLUS_2( oITER);
  for( int i = 2; oITER_N_PLUS_2 != roDIGIS.end() && i > 0; --i) {
  ++oITER_N_PLUS_2;
  }

  // Check if there is no N-2/N+2 Digi specified
  if( ( ( oITER_N_MINUS_2 == roDIGIS.end() ||
  oITER_N_MINUS_2->strip() + 1 != ( oITER_N_MINUS_2 + 1)->strip()) &&
  ( oITER_N_PLUS_2 == roDIGIS.end() ||
  oITER_N_PLUS_2->strip() - 1 != ( oITER_N_PLUS_2 - 1)->strip())) ) {
  dClusterCrossTalk = calculateClusterCrossTalk( roSTRIP_AMPLITUDES[0],
  roSTRIP_AMPLITUDES[1],
  roSTRIP_AMPLITUDES[2]);
  }
  }
  }
  default:
  break;
  }



  return dClusterCrossTalk;
  }
*/
