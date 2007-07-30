/////////////////////////////////////////////////////////////
//
// Code adapted from MTCCNtupleMaker 
//
// Modifications and rationalization of ntuple structure:
//
// Marco de Mattia, Tommaso Dorigo, Mia Tosi - February 2007
//
/////////////////////////////////////////////////////////////

#include <memory>
#include <string>
#include <iostream>
#include <fstream>

// Definitions 
// -----------
#include "AnalysisExamples/SiStripDetectorPerformance/interface/TIFNtupleMaker.h"
// General stuff
// -------------
#include "FWCore/Framework/interface/Handle.h"
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
// For trackinfo:
// --------------
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoTrackAssociation.h"
// From the example:
// -----------------
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoTrackAssociation.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/ESHandle.h"

using namespace std;

// Constructor
// -----------
TIFNtupleMaker::TIFNtupleMaker(edm::ParameterSet const& conf) : 
  conf_                     (conf), 
  filename_                 (conf.getParameter<std::string>         ("fileName")                 ),
  oSiStripDigisLabel_       (conf.getUntrackedParameter<std::string>("oSiStripDigisLabel")       ),
  oSiStripDigisProdInstName_(conf.getUntrackedParameter<std::string>("oSiStripDigisProdInstName")),
  dCROSS_TALK_ERR           (conf.getUntrackedParameter<double>     ("dCrossTalkErr")            ),

  bTriggerDT(false),
  bTriggerCSC(false),
  bTriggerRBC1(false),
  bTriggerRBC2(false),
  bTriggerRPC(false) {
  
    Anglefinder = new TrackLocalAngleTIF();  
  }

void TIFNtupleMaker::beginJob(const edm::EventSetup& c) {

  hFile = new TFile (filename_.c_str(), "RECREATE" );
  
  // Main tree on hits
  // -----------------
  TIFNtupleMakerTree = new TTree ( "TIFNtupleMakerTree" , "This is a cluster variables tree");
  TIFNtupleMakerTree->Branch("run",                  &run,                  "run/I"                  );
  TIFNtupleMakerTree->Branch("event",                &event,                "event/I"                );
  TIFNtupleMakerTree->Branch("eventcounter",         &eventcounter,         "eventcounter/I"         );
  TIFNtupleMakerTree->Branch("module",               &module,               "module/I"               );
  TIFNtupleMakerTree->Branch("type",                 &type,                 "type/I"                 );
  TIFNtupleMakerTree->Branch("layer",                &layer,                "layer/I"                );
  TIFNtupleMakerTree->Branch("string",               &string,               "string/I"               );
  TIFNtupleMakerTree->Branch("rod",                  &rod,                  "rod/I"                  );
  TIFNtupleMakerTree->Branch("extint",               &extint,               "extint/I"               );
  TIFNtupleMakerTree->Branch("size",                 &size,                 "size/I"                 );
  TIFNtupleMakerTree->Branch("angle",                &angle,                "angle/F"                );
  TIFNtupleMakerTree->Branch("tk_phi",               &tk_phi,               "tk_phi/F"               );
  TIFNtupleMakerTree->Branch("tk_theta",             &tk_theta,             "tk_theta/F"             );
  TIFNtupleMakerTree->Branch("tk_id",                &tk_id,                "tk_id/I"                );
  TIFNtupleMakerTree->Branch("shared",               &shared,               "shared/O"               );
  TIFNtupleMakerTree->Branch("sign",                 &sign,                 "sign/I"                 );
  TIFNtupleMakerTree->Branch("bwfw",                 &bwfw,                 "bwfw/I"                 );
  TIFNtupleMakerTree->Branch("wheel",                &wheel,                "wheel/I"                );
  TIFNtupleMakerTree->Branch("monostereo"  ,         &monostereo,           "monostereo/I"           );
  TIFNtupleMakerTree->Branch("stereocorrection",     &stereocorrection,     "stereocorrection/F"     );
  TIFNtupleMakerTree->Branch("localmagfield",        &localmagfield,        "localmagfield/F"        );
  TIFNtupleMakerTree->Branch("momentum",             &momentum,             "momentum/F"             );
  TIFNtupleMakerTree->Branch("pt",                   &pt,                   "pt/F"                   );
  TIFNtupleMakerTree->Branch("charge",               &charge,               "charge/I"               );
  TIFNtupleMakerTree->Branch("eta",                  &eta,                  "eta/F"                  );
  TIFNtupleMakerTree->Branch("phi",                  &phi,                  "phi/F"                  );
  TIFNtupleMakerTree->Branch("hitspertrack"  ,       &hitspertrack,         "hitspertrack/I"         );
  TIFNtupleMakerTree->Branch("normchi2",             &normchi2,             "normchi2/F"             );
  TIFNtupleMakerTree->Branch("chi2",                 &chi2,                 "chi2/F"                 );
  TIFNtupleMakerTree->Branch("ndof",                 &ndof,                 "ndof/F"                 );
  TIFNtupleMakerTree->Branch("numberoftracks",       &numberoftracks,       "numberoftracks/I"       );
  TIFNtupleMakerTree->Branch("trackcounter",         &trackcounter,         "trackcounter/I"         );
  TIFNtupleMakerTree->Branch("clusterpos",           &clusterpos,           "clusterpos/F"           );
  TIFNtupleMakerTree->Branch("clustereta",           &clustereta,           "clustereta/F"           );
  TIFNtupleMakerTree->Branch("clusterchg",           &clusterchg,           "clusterchg/F"           );
  TIFNtupleMakerTree->Branch("clusterchgl",          &clusterchgl,          "clusterchgl/F"          );
  TIFNtupleMakerTree->Branch("clusterchgr",          &clusterchgr,          "clusterchgr/F"          );
  TIFNtupleMakerTree->Branch("clusternoise",         &clusternoise,         "clusternoise/F"         );
  TIFNtupleMakerTree->Branch("clusterbarycenter",    &clusterbarycenter,    "clusterbarycenter/F"    );
  TIFNtupleMakerTree->Branch("clustermaxchg",        &clustermaxchg,        "clustermaxchg/F"        );
  TIFNtupleMakerTree->Branch("clusterseednoise",     &clusterseednoise,     "clusterseednoise/F"     );
  TIFNtupleMakerTree->Branch("clustercrosstalk",     &clustercrosstalk,     "clustercrosstalk/F"     );
  TIFNtupleMakerTree->Branch("localPositionX",       &dLclX,                "dLclX/F"                );
  TIFNtupleMakerTree->Branch("localPositionY",       &dLclY,                "dLclY/F"                );
  TIFNtupleMakerTree->Branch("localPositionZ",       &dLclZ,                "dLclZ/F"                );
  TIFNtupleMakerTree->Branch("globalPositionX",      &dGlbX,                "dGlbX/F"                );
  TIFNtupleMakerTree->Branch("globalPositionY",      &dGlbY,                "dGlbY/F"                );
  TIFNtupleMakerTree->Branch("globalPositionZ",      &dGlbZ,                "dGlbZ/F"                );
  TIFNtupleMakerTree->Branch("numberofclusters",     &numberofclusters,     "numberofclusters/I"     );
  TIFNtupleMakerTree->Branch("numberoftkclusters",   &numberoftkclusters,   "numberoftkclusters/I"   );
  TIFNtupleMakerTree->Branch("numberofnontkclusters",&numberofnontkclusters,"numberofnontkclusters/I");

  // Track tree
  // ----------
  poTrackTree = new TTree ( "TrackTree", "This is a track variables tree" );
  poTrackTree->Branch("run",                  &run,                  "run/I"                  );
  poTrackTree->Branch("event",                &event,                "event/I"                );
  poTrackTree->Branch("eventcounter",         &eventcounter,         "eventcounter/I"         );
  poTrackTree->Branch("momentum",             &momentum,             "momentum/F"             );
  poTrackTree->Branch("pt",                   &pt,                   "pt/F"                   );
  poTrackTree->Branch("charge",               &charge,               "charge/I"               );
  poTrackTree->Branch("eta",                  &eta,                  "eta/F"                  );
  poTrackTree->Branch("phi",                  &phi,                  "phi/F"                  );
  poTrackTree->Branch("hitspertrack",         &hitspertrack,         "hitspertrack/I"         );
  poTrackTree->Branch("normchi2",             &normchi2,             "normchi2/F"             );
  poTrackTree->Branch("chi2",                 &chi2,                 "chi2/F"                 );
  poTrackTree->Branch("ndof",                 &ndof,                 "ndof/F"                 );
  poTrackTree->Branch("numberoftracks",       &numberoftracks,       "numberoftracks/I"       );
  poTrackTree->Branch("trackcounter",         &trackcounter,         "trackcounter/I"         );
  poTrackTree->Branch("numberofclusters",     &numberofclusters,     "numberofclusters/I"     );
  //  poTrackTree->Branch("numberoftkclusters",   &numberoftkclusters,   "numberoftkclusters/I"   );
  //  poTrackTree->Branch("numberofnontkclusters",&numberofnontkclusters,"numberofnontkclusters/I");

  // New tree for number of tracks
  // -----------------------------
  poTrackNum = new TTree ( "TrackNum", "This is an event tree containing the number of objects" );
  poTrackNum->Branch("numberoftracks",       &numberoftracks,       "numberoftracks/I"       );
  poTrackNum->Branch("numberofclusters",     &numberofclusters,     "numberofclusters/I"     );
  poTrackNum->Branch("numberoftkclusters",   &numberoftkclusters,   "numberoftkclusters/I"   );
  poTrackNum->Branch("numberofnontkclusters",&numberofnontkclusters,"numberofnontkclusters/I");

  // Global counters
  // ---------------
  eventcounter = 0;
  trackcounter = 0;
  
  edm::ESHandle<MagneticField> esmagfield;
  c.get<IdealMagneticFieldRecord>().get(esmagfield);
  magfield=&(*esmagfield);
    
  edm::ESHandle<TrackerGeometry> estracker;
  c.get<TrackerDigiGeometryRecord>().get(estracker);
  tracker=&(*estracker); 

}

// Virtual destructor needed
// -------------------------

TIFNtupleMaker::~TIFNtupleMaker() {  
  // delete poTrackTree;
}  

// Analyzer: Functions that gets called by framework every event
// -------------------------------------------------------------
void TIFNtupleMaker::analyze(const edm::Event& e, const edm::EventSetup& es) {

  using namespace edm;

  std::map<std::pair<uint32_t, int>, int>           oProcessedClusters;
  std::map<SiSubDet, std::map<unsigned char, int> > oClustersPerLayer;
  std::vector<SiStripDigi>                          oDigis;
  std::vector<SiStripClusterInfo>                   oClusterInfos;

  // Step A: Get Inputs   
  // Initialize the angle finder
  // ---------------------------
  Anglefinder->init(es);

  // TrackCollection
  // -------------------
  edm::Handle<reco::TrackCollection> trackCollection;
  e.getByLabel( conf_.getParameter<std::string>( "TracksLabel"), 
		trackCollection);

  // Trackinfocollection --> 7/2/2007
  // --------------------------------
  edm::InputTag TkiTag = conf_.getParameter<edm::InputTag>("TrackInfoLabel");
  edm::Handle<reco::TrackInfoTrackAssociationCollection> TItkassociatorCollection;
  e.getByLabel(TkiTag,TItkassociatorCollection);

  // DetSetVector SiStripClusterInfos
  // --------------------------------
  edm::Handle<edm::DetSetVector<SiStripClusterInfo> > oDSVClusterInfos;
  e.getByLabel( "siStripClusterInfoProducer", oDSVClusterInfos);

  // DetSetVector SiStripDigis
  // -------------------------
  edm::Handle<edm::DetSetVector<SiStripDigi> > oDSVDigis;

  // Take the Digis
  // --------------
  std::cout << "oSiStripDigisProdInstName_.size() = " 
	    << oSiStripDigisProdInstName_.size() << std::endl;
  if( oSiStripDigisProdInstName_.size()) {
    std::cout << "if" << std::endl;
    e.getByLabel( oSiStripDigisLabel_.c_str(), 
		  oSiStripDigisProdInstName_.c_str(), oDSVDigis);
  } else {
    std::cout << "else" << std::endl;
    e.getByLabel( oSiStripDigisLabel_.c_str(), oDSVDigis);
  }
  //   std::cout << "numberOfCLusters:= oDSVDigis->size() = " << oDSVDigis->size() << std::endl;

  // Take the number of cluster in the event
  // ---------------------------------------
  // ---------------------
  // CHECK numberofcluster
  // ---------------------

  // Initialize
  numberofclusters = 0;

  edm::DetSetVector<SiStripClusterInfo>::const_iterator clusterNumIter;
  for( clusterNumIter = oDSVClusterInfos->begin(); clusterNumIter != oDSVClusterInfos->end(); ++clusterNumIter ){
    std::cout << "clusterNumIter->data.size() = " << clusterNumIter->data.size() << std::endl;
    numberofclusters += clusterNumIter->data.size();
    std::cout << "numberofclusters = " << numberofclusters << std::endl;
  }

  //  numberofclusters = oDSVDigis->size();

  // TrackCollection
  // ---------------
  const reco::TrackCollection *tracks=trackCollection.product();
  // Take the number of tracks in the event
  // --------------------------------------
  numberoftracks = tracks->size();


  // Initializations per event
  // -------------------------
  nTrackClusters = 0;
  run       = e.id().run();
  event     = e.id().event();
  
  eventcounter++;
  std::cout << "Event number " << eventcounter << std::endl;

  if ( numberoftracks > 0 ){
    int nTrackNum = 0;      
    // Loop on tracks
    // --------------
    std::cout << "Starting to loop on all the track collection" << std::endl;
    reco::TrackCollection::const_iterator tkIter;
    for( tkIter=trackCollection->begin(); tkIter!=trackCollection->end(); ++tkIter ){

      // Initializations per track
      // -------------------------
      tk_id                 =   0;    
      momentum              = -99;  
      pt	            = -99;  
      charge                = -99;  
      eta	            = -99;  
      phi	            = -99;  
      hitspertrack          = -99;  
      normchi2              = -99;  
      chi2                  = -99;  
      ndof                  = -99;  

      reco::TrackRef trackref=reco::TrackRef(trackCollection,nTrackNum);
      std::cout << "Track pt" <<trackref->pt() << std::endl;
      reco::TrackInfoRef trackinforef = (*TItkassociatorCollection.product())[trackref];

      // The track seed:
      // ---------------
      const TrajectorySeed seed = trackinforef->seed();
      std::cout << "Test trackinfores seeds = " << seed.nHits()          << std::endl;
      std::cout << " N hits on Track = "        << (*tkIter).recHitsSize() << std::endl;


      ++nTrackNum;
      ++trackcounter;  // only count processed tracks (not tracks->size())

      // Set the id here, so that tk_id = 0 if not on track 
      //                            and > 0 for track index
      // --------------------------------------------------
      tk_id        = nTrackNum;
      momentum     = tkIter->p();
      pt           = tkIter->pt();
      charge       = tkIter->charge();
      eta          = tkIter->eta();
      phi          = tkIter->phi();
      hitspertrack = tkIter->recHitsSize();
      normchi2     = tkIter->normalizedChi2();
      chi2         = tkIter->chi2();
      ndof         = tkIter->ndof();
      
      // Disentangle matched rechits and evaluate corresponding 
      // local directions and angles
      // The method requires only the trackinfo.
      // ---------------------------------------      
      std::vector<std::pair<const TrackingRecHit *,float> > hitangle;
      std::auto_ptr<std::vector<std::pair<const TrackingRecHit *,float> > > hitangleXZ;
      std::auto_ptr<std::vector<std::pair<const TrackingRecHit *,float> > > hitangleYZ;
      hitangle   = Anglefinder->SeparateHits(trackinforef);
      hitangleXZ = Anglefinder->getXZHitAngle();
      hitangleYZ = Anglefinder->getYZHitAngle();

      int nHitNum = 0;
      // Loop on hits belonging to this track
      // ------------------------------------
      std::cout << "Starting to loop on all the track hits" << std::endl;
      std::vector<std::pair<const TrackingRecHit *,float> >::iterator hitsIter;
      for( hitsIter=hitangle.begin(); hitsIter!=hitangle.end(); ++hitsIter ){
	
	TrackLocalAngleTIF::HitAngleAssociation::reference hitsrefXZ = (*hitangleXZ)[nHitNum];
	TrackLocalAngleTIF::HitAngleAssociation::reference hitsrefYZ = (*hitangleYZ)[nHitNum];
	 
	// Initializations per hit
	// -----------------------
	module                = -99;
	type                  = -99;
	layer                 = -99;      // There is no layer for TID and TEC
	string                = -99;      // String exists only for TIB
	rod                   = -99;      // Only for TOB
	extint                = -99;      // Olny for TIB
	size                  = -99;
	shared                = false;    // filled Only for hits belonging to track
	angle                 = -9999;    // filled Only for hits belonging to track
	tk_phi                = -9999;    // filled Only for hits belonging to track
	tk_theta              = -9999;    // filled Only for hits belonging to track
	sign                  = -99;      // Not needed
	bwfw                  = -99;
	wheel                 = -99;      // Only for TID, TEC
	monostereo            = -99; 
	stereocorrection      = -9999;    // filled Only for hits belonging to track
	localmagfield         = -99.;
	clusterpos            = -99.;  
	clustereta            = -99.;
	clusterchg            = -99.;
	clusterchgl           = -99.;
	clusterchgr           = -99.;
	clusternoise          = -99;
	clusterbarycenter     = -99;
	clustermaxchg         = -99.;
	clusterseednoise      = -99;
	clustercrosstalk      = -99.;
	dLclX	              = -99;
	dLclY	              = -99;
	dLclZ	              = -99;
	dGlbX	              = -99;
	dGlbY	              = -99;
	dGlbZ	              = -99;
	numberoftkclusters    = -99;
	numberofnontkclusters = -99;
	
	
	// Taking the hits previously separated by TrackLocalAngleTIF
	// ----------------------------------------------------------
	const SiStripRecHit2D* hit = dynamic_cast<const SiStripRecHit2D*>(hitsIter->first);
	dLclX = hit->localPosition().x();
	dLclY = hit->localPosition().y();
	dLclZ = hit->localPosition().z();

	GlobalPoint oRecHitGlobalPos = tracker->idToDet
	  ( hit->geographicalId())->toGlobal( hit->localPosition());
	dGlbX = oRecHitGlobalPos.x();
	dGlbY = oRecHitGlobalPos.y();
	dGlbZ = oRecHitGlobalPos.z();

	// Taking the clusters corresponding to the hits
	// ---------------------------------------------
	const edm::Ref<edm::DetSetVector<SiStripCluster>,
	  SiStripCluster,
	  edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster = hit->cluster();

	std::cout << "cluster->geographicalId() = " 
		  << cluster->geographicalId() << std::endl;

	oDigis        = oDSVDigis->operator[](cluster->geographicalId()).data;
	oClusterInfos = oDSVClusterInfos->operator[](cluster->geographicalId()).data;

	// Loop on ClusterInfo
	// -------------------
	std::cout << "Starting to loop on ClusterInfo" << std::endl;
	std::vector<SiStripClusterInfo>::iterator oIter;
	for( oIter = oClusterInfos.begin(); oIter != oClusterInfos.end(); ++oIter){
	
	  // ClusterInfo matched given cluster
	  // ---------------------------------
	  if( oIter->firstStrip() == 
	      cluster->firstStrip()) {
	    
	    clusterpos    = oIter->position();
	    clusterchg    = oIter->charge();
	    size          = (int)oIter->width();
	    clusterchgl   = oIter->chargeL();
	    clusterchgr   = oIter->chargeR();
	    clusternoise  = oIter->noise();
	    clustermaxchg = oIter->maxCharge();

	    std::cout << "Cluster Info size = "	    << oIter->stripAmplitudes().size() 
		      << " or  width "              << oIter->width() << std::endl;
	    std::cout << "Cluster Info Position = " << clusterpos     << std::endl;
	    std::cout << "Cluster Info Charge = "   << clusterchg     << std::endl;

	    // Fill here using clusterinfo thing
	    // ---------------------------------
	    clustereta        = getClusterEta( oIter->stripAmplitudes(),
					       oIter->firstStrip(),
					       oDigis);
	    clustercrosstalk  = getClusterCrossTalk( oIter->stripAmplitudes(),
						     oIter->firstStrip(),
						     oDigis);
	    clusterbarycenter = cluster->barycenter();
	    clusterseednoise  = oIter->stripNoises().operator[]( (int)cluster->barycenter() -
								 cluster->firstStrip()        );

	    // Mark Current ClusterInfo as processed
	    // [Note: One Module might hold several clusters (!)]

	    // To select which clusters belong to more than one track
	    // ------------------------------------------------------
	    std::pair<std::map<pair<uint32_t, int>,int>::iterator, bool> inserted;
	    inserted = oProcessedClusters.insert(std::make_pair( std::make_pair(cluster->geographicalId(),
								      oIter->firstStrip()), 0) );
	    shared = false;

	    // Write if the cluster is shared
	    // ------------------------------
	    if ( !(inserted.second) ) {
	      shared = true;
	    }
	    break;
	  }
	  // ClusterInfo was found but didn't match given cluster
	  // ----------------------------------------------------
	  else {
	  }
	} // end clusterInfo loop

	if( !shared ) ++nHitNum;
	
	StripSubdetector oStripSubdet = (StripSubdetector)hit->geographicalId();

	type       = oStripSubdet.subdetId();
	std::cout << "type = " << type << std::endl;
	monostereo = oStripSubdet.stereo();

	// Repeated later
	// maybe create a method
	switch( type) {
	case StripSubdetector::TIB:
	  {
	    if(conf_.getParameter<bool>("TIB_ON")){
	      TIBDetId oTIBDetId( oStripSubdet);
	      layer  = oTIBDetId.layer();
	      bwfw   = oTIBDetId.string()[0];
	      extint = oTIBDetId.string()[1];
	      string = oTIBDetId.string()[2];
	    }
	    break;
	  }
	case StripSubdetector::TID:
	  {
	    if(conf_.getParameter<bool>("TID_ON")){
	      TIDDetId oTIDDetId( oStripSubdet);
	      wheel = oTIDDetId.wheel();
	      bwfw  = oTIDDetId.module()[0];
	    }
	    break;
	  }
	case StripSubdetector::TOB:
	  {
	    if(conf_.getParameter<bool>("TOB_ON")) {
	      TOBDetId oTOBDetId( oStripSubdet);
	      layer = oTOBDetId.layer();
	      bwfw  = oTOBDetId.rod()[0];
	      rod   = oTOBDetId.rod()[1];
	    }
	    break;
	  }
	case StripSubdetector::TEC:
	  {
	    if(conf_.getParameter<bool>("TEC_ON")) {
	      TECDetId oTECDetId( oStripSubdet);
	      wheel = oTECDetId.wheel();
	      bwfw  = oTECDetId.petal()[0];
	    }
	    break;
	  }
	}

	module = (hit->geographicalId()).rawId();
      
	angle    = hitsIter->second;
	tk_phi   = hitsrefXZ.second;
	tk_theta = hitsrefYZ.second;
	float angleXZ = hitsrefXZ.second;
	float angleYZ = hitsrefYZ.second;

	    
	// Local Magnetic Field
	// --------------------    
	const GeomDet *geomdet = tracker->idToDet(hit->geographicalId());
	LocalPoint localp(0,0,0);
	const GlobalPoint globalp = (geomdet->surface()).toGlobal(localp);
	GlobalVector globalmagdir = magfield->inTesla(globalp);
	localmagdir   = (geomdet->surface()).toLocal(globalmagdir);
	localmagfield = localmagdir.mag();
	    
	if( localmagfield != 0. ){
	  // ---------------
	  // CHECK THIS PART
	  // ---------------
	  // Sign correction for TIB and TOB
	  // -------------------------------
	  if( (oStripSubdet.subdetId() ==
	       int (StripSubdetector::TIB) ) ||
	      (oStripSubdet.subdetId() ==
	       int (StripSubdetector::TOB) )    ){
	      
	    LocalVector ylocal(0,1,0);
	      
	    float normprojection = (localmagdir * ylocal)/(localmagfield);
	      
	    if(normprojection>0){sign =  1;}
	    if(normprojection<0){sign = -1;}
		
	    // Stereocorrection applied in TrackLocalAngleTIF
	    // ----------------------------------------------
	    if( (oStripSubdet.stereo() == 1 ) && 
		(normprojection == 0.)    ){
	      LogDebug("TIFNtupleMaker::analyze") << "Error: TIB|TOB YBprojection = 0";
	    }
	    if( (oStripSubdet.stereo() == 1 ) && 
		(normprojection != 0.)    ){
	      stereocorrection = 1/normprojection;
	      stereocorrection*=sign;
		
	      float tg = tan((angle*TMath::Pi())/180);
	      tg*=stereocorrection;
	      angle = atan(tg)*180/TMath::Pi();
		
	      tg = tan(angleXZ);
	      tg*=stereocorrection;
	      angleXZ = atan(tg);
		
	      tg = tan(angleYZ);
	      tg*=stereocorrection;
	      angleYZ = atan(tg);
	    }
	      
	    angle   *= sign;
	    angleXZ *= sign;
	    angleYZ *= sign;       
	  }
	} // end if localmagfield!=0

	// Filling Main Tree on hits belonging to track
	// --------------------------------------------  
	TIFNtupleMakerTree->Fill();

	LogDebug("TIFNtupleMaker::analyze") << "Tree Filled";

      } // end hitsIter loop

      nTrackClusters += nHitNum;

      // Fill track tree
      // ---------------
      // ----- WARNING: variables unfilled -----
      // - numberoftkclusters   
      // - numberofnontkclusters
      poTrackTree->Fill();
      
    } // end loop on tracks
  } // end if tracks.size() > 0

  

  // ----------------------------------------------------
  // Now work out all Cluster that were left unprocessed,
  // thus do not belong to tracks
  // ----------------------------------------------------
  // They don't belong to any track
  tk_id = 0;
  // Loop over DetSetVector ClusterInfo 
  // ----------------------------------
  edm::DetSetVector<SiStripClusterInfo>::const_iterator oDSVIter;
  for( oDSVIter = oDSVClusterInfos->begin(); oDSVIter != oDSVClusterInfos->end(); ++oDSVIter ){
    std::cout << "inside loop over modules" << std::endl;

    StripSubdetector oStripSubdet( oDSVIter->id );
    unsigned char ucLayer;
    switch( oStripSubdet.subdetId() ) {
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
    std::cout << "oDSVIter->size() = " << oDSVIter->size() << std::endl;

    // Extract ClusterInfos collection for given module
    oClusterInfos = oDSVIter->data;
    oDigis        = oDSVDigis->operator[]( oDSVIter->id).data;
 
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

    std::cout << "clusterinfo size = " << oClusterInfos.size() << std::endl;

    numberoftkclusters = nTrackClusters;
    numberofnontkclusters = numberofclusters - nTrackClusters;

    // Loop over ClusterInfos collection
    // ---------------------------------
    std::vector<SiStripClusterInfo>::iterator oIter;
    for( oIter= oClusterInfos.begin(); oIter != oClusterInfos.end();	++oIter ){

      // 	// Check if given ClusterInfo was already processed
      // 	// ------------------------------------------------
      if( oProcessedClusters.end() ==
	  oProcessedClusters.find( std::make_pair( oDSVIter->id, oIter->firstStrip() ) ) ) {

	// Initializations per hit
	// -----------------------
	module                = -99;
	type                  = -99;
	layer                 = -99;      // There is no layer for TID and TEC
	string                = -99;      // String exists only for TIB
	rod                   = -99;      // Only for TOB
	extint                = -99;      // Olny for TIB
	size                  = -99;
	angle                 = -9999;    // filled Only for hits belonging to track
	tk_phi                = -9999;    // filled Only for hits belonging to track
	tk_theta              = -9999;    // filled Only for hits belonging to track
	tk_id                 = 0;        // Only for track
	shared                = false;    // Only for track
	momentum              = -99;      // Only for track
	pt	                = -99;      // Only for track
	charge                = -99;      // Only for track
	eta	                = -99;      // Only for track
	phi	                = -99;      // Only for track
	hitspertrack          = -99;      // Only for track
	normchi2              = -99;      // Only for track
	chi2                  = -99;      // Only for track
	ndof                  = -99;      // Only for track
	sign                  = -99;      // Not needed
	bwfw                  = -99;
	wheel                 = -99;      // Only for TID, TEC
	monostereo            = -99; 
	stereocorrection      = -9999;    // Only for track
	localmagfield         = -99.;
	clusterpos            = -99.;  
	clustereta            = -99.;
	clusterchg            = -99.;
	clusterchgl           = -99.;
	clusterchgr           = -99.;
	clusternoise          = -99;
	clusterbarycenter     = -99;
	clustermaxchg         = -99.;
	clusterseednoise      = -99;
	clustercrosstalk      = -99.;
	dLclX	                = -99;
	dLclY	                = -99;
	dLclZ	                = -99;
	dGlbX	                = -99;
	dGlbY	                = -99;
	dGlbZ	                = -99;
	numberoftkclusters    = -99;   // could be filled before?
	numberofnontkclusters = -99;


	StripSubdetector oStripSubdet( oIter->geographicalId());
	module     = oStripSubdet.rawId();
	type	     = oStripSubdet.subdetId();
	monostereo = oStripSubdet.stereo();

	switch( type) {
	case StripSubdetector::TIB:
	  {
	    if(conf_.getParameter<bool>("TIB_ON")){
	      TIBDetId oTIBDetId( oStripSubdet);
	      layer  = oTIBDetId.layer();
	      bwfw   = oTIBDetId.string()[0];
	      extint = oTIBDetId.string()[1];
	      string = oTIBDetId.string()[2];
	    }
	    break;
	  }
	case StripSubdetector::TID:
	  {
	    if(conf_.getParameter<bool>("TID_ON")){
	      TIDDetId oTIDDetId( oStripSubdet);
	      wheel = oTIDDetId.wheel();
	      bwfw  = oTIDDetId.module()[0];
	    }
	    break;
	  }
	case StripSubdetector::TOB:
	  {
	    if(conf_.getParameter<bool>("TOB_ON")) {
	      TOBDetId oTOBDetId( oStripSubdet);
	      layer = oTOBDetId.layer();
	      bwfw  = oTOBDetId.rod()[0];
	      rod   = oTOBDetId.rod()[1];
	    }
	    break;
	  }
	case StripSubdetector::TEC:
	  {
	    if(conf_.getParameter<bool>("TEC_ON")) {
	      TECDetId oTECDetId( oStripSubdet);
	      wheel = oTECDetId.wheel();
	      bwfw  = oTECDetId.petal()[0];
	    }
	    break;
	  }
	}

	// ClusterInfo was not processed yet
	// ---------------------------------
	size             = (int)oIter->width(); // cluster width
	clusterpos       = oIter->position();
	clusterchg       = oIter->charge();
	clusterchgl      = oIter->chargeL();
	clusterchgr      = oIter->chargeR();
	clusternoise     = oIter->noise();
	clustermaxchg    = oIter->maxCharge();
	clustereta       = getClusterEta( oIter->stripAmplitudes(), 
					  oIter->firstStrip(),
					  oDigis);
	clustercrosstalk = getClusterCrossTalk( oIter->stripAmplitudes(),
						oIter->firstStrip(),
						oDigis);

	// Filling Main Tree on hits not belonging to track
	// ------------------------------------------------  
	TIFNtupleMakerTree->Fill();
      } // if was not already processed
    } // end loop on clusterinfo in detset
  } // end loop on detsetvector
  std::cout << "end of loop over modules" << std::endl;

  // Work in progress
  // ----------------
  // -----------------------------------
  // WARNING:
  // nTotClusters does not exist anymore
  // it is replaced by numberofclusters
  // -----------------------------------
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

  //sistemare e add TID


  // Fill tracks number Tree
  // -----------------------
  poTrackNum->Fill();

} // end analyze


//EndJob
void TIFNtupleMaker::endJob(){

  std::cout << "endJob"                                 << std::endl;
  std::cout << ">>> TOTAL EVENT = "     << eventcounter << std::endl;
  std::cout << ">>> NUMBER OF TRACKS = "<< trackcounter << std::endl;


  // Save Tree in ROOT file
  // ----------------------
  hFile->Write();
  hFile->Close();
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
  // -----------------------
  int anMaxSignal[2][2];

  // Null array before using it
  for( int i = 0; i < 2; ++i) {
    for( int j = 0; j < 2; ++j) {
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
