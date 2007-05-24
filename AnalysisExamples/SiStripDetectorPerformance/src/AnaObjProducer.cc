//#define DEBUG

#include <memory>
#include <string>
#include <iostream>
#include <fstream>

// Definitions 
// -----------
#include "AnalysisExamples/SiStripDetectorPerformance/interface/AnaObjProducer.h"
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
#include "FWCore/Framework/interface/EDProducer.h"
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
using namespace anaobj;

// Constructor
// -----------
AnaObjProducer::AnaObjProducer(edm::ParameterSet const& conf) : 
  conf_                     (conf), 
  filename_                 (conf.getParameter<std::string>            ("fileName")                 ),
  oSiStripDigisLabel_       (conf.getUntrackedParameter<std::string>   ("oSiStripDigisLabel")       ),
  oSiStripDigisProdInstName_(conf.getUntrackedParameter<std::string>   ("oSiStripDigisProdInstName")),
  analyzedtrack_            (conf.getParameter<std::string>            ("analyzedtrack")            ),
  analyzedcluster_          (conf.getParameter<std::string>            ("analyzedcluster")          ),
  dCROSS_TALK_ERR           (conf.getUntrackedParameter<double>        ("dCrossTalkErr")            ) {
  
  Anglefinder = new TrackLocalAngleTIF();  

#ifdef DEBUG
  std::cout << "analyzedtrack_ = " << analyzedtrack_ << std::endl;
  std::cout << "analyzedcluster_ = " << analyzedcluster_ << std::endl;
#endif

  produces<AnalyzedTrackCollection>( analyzedtrack_ );
  produces<AnalyzedClusterCollection>( analyzedcluster_ );
}


// BeginJob
// --------
void AnaObjProducer::beginJob(const edm::EventSetup& es) {

  // Global counters
  // ---------------
  eventcounter   = 0;
  trackcounter   = 0;
  clustercounter = 0;
  clusterTKcounter = 0;
  clusterNNTKcounter = 0;
  
  edm::ESHandle<MagneticField> esmagfield;
  es.get<IdealMagneticFieldRecord>().get(esmagfield);
  magfield=&(*esmagfield);
    
  edm::ESHandle<TrackerGeometry> estracker;
  es.get<TrackerDigiGeometryRecord>().get(estracker);
  tracker=&(*estracker);

}

// Virtual destructor needed
// -------------------------

AnaObjProducer::~AnaObjProducer() {
  delete Anglefinder;
}

// Producer: Function that gets called by framework every event
// ------------------------------------------------------------
void AnaObjProducer::produce(edm::Event& e, const edm::EventSetup& es) {

  using namespace edm;

  std::vector<SiStripDigi>                          oDigis;
  std::vector<SiStripClusterInfo>                   oClusterInfos;

  // Vector to store the AnalyzedCluster structs:
  std::auto_ptr<AnalyzedClusterCollection> v_anaclu_ptr(new AnalyzedClusterCollection);

  // Map to store the pair needed to identify the cluster with the AnalyzedCluster
  // and the index of the AnalyzedCluster stored in the vector (to have access to it later)
  clustermap                                        map_analyzedCluster;

  // Vector to store the AnalyzedTrack structs
  std::auto_ptr<AnalyzedTrackCollection> v_anatk_ptr(new AnalyzedTrackCollection);
  // Map to store the pair needed to identify the track with the AnalyzedTrack
  // and the index of the AnalyzedTrack stored in the vector (to have access to it later)
  trackmap map_analyzedTrack;

  // Step 0: Declare Ref and RefProd
  AnalyzedClusterRefProd anaclu_refprod = e.getRefBeforePut<AnalyzedClusterCollection>();
  AnalyzedTrackRefProd anatk_refprod = e.getRefBeforePut<AnalyzedTrackCollection>();
  // Declare key_type (counter for the Ref)
  AnalyzedClusterRef::key_type anaclu_id = 0;
  AnalyzedTrackRef::key_type anatk_id = 0;

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

  edm::Handle<edm::DetSetVector<SiStripCluster> > oDSVCluster;
  e.getByLabel( "siStripClusters", oDSVCluster);    

  // DetSetVector SiStripDigis
  // -------------------------
  edm::Handle<edm::DetSetVector<SiStripDigi> > oDSVDigis;

  // Take the Digis
  // --------------
#ifdef DEBUG
  std::cout << "oSiStripDigisProdInstName_.size() = " 
	    << oSiStripDigisProdInstName_.size() << std::endl;
#endif
  if( oSiStripDigisProdInstName_.size()) {
#ifdef DEBUG
    std::cout << "if" << std::endl;
#endif
    e.getByLabel( oSiStripDigisLabel_.c_str(), 
		  oSiStripDigisProdInstName_.c_str(), oDSVDigis);
  } else {
#ifdef DEBUG
    std::cout << "else" << std::endl;
#endif
    e.getByLabel( oSiStripDigisLabel_.c_str(), oDSVDigis);
  }

  // Take the number of cluster in the event
  // ---------------------------------------
  // Initialize
  // ----------
  numberofclusters = 0;
  edm::DetSetVector<SiStripClusterInfo>::const_iterator clusterNumIter;
  for( clusterNumIter = oDSVClusterInfos->begin();
       clusterNumIter != oDSVClusterInfos->end();
       ++clusterNumIter ){
    numberofclusters += clusterNumIter->data.size();
#ifdef DEBUG
    std::cout << "clusterNumIter->data.size() = " << clusterNumIter->data.size() << std::endl;
    std::cout << "numberofclusters = " << numberofclusters << std::endl;
#endif
  }

  // TrackCollection
  // ---------------
  const reco::TrackCollection *tracks=trackCollection.product();
  // Take the number of tracks in the event
  // --------------------------------------
  numberoftracks = tracks->size();


  // Initializations per event
  // -------------------------
  countOn  = 0;
  countOff = 0;
  countAll = 0;
  nTotClustersTIB = 0;
  nTotClustersTID = 0;
  nTotClustersTOB = 0;
  nTotClustersTEC = 0;
  nTrackClusters  = 0;
  run       = e.id().run();
  event     = e.id().event();
  
  eventcounter++;
#ifdef DEBUG
  std::cout << "Event number " << eventcounter << std::endl;
#endif
  
  // Perform Cluster Study (irrespectively to tracks)
  // ------------------------------------------------

  // Loop over DetSetVector ClusterInfo
  // ----------------------------------
  edm::DetSetVector<SiStripClusterInfo>::const_iterator oDSVclusterInfoIter;
  for( oDSVclusterInfoIter = oDSVClusterInfos->begin();
       oDSVclusterInfoIter != oDSVClusterInfos->end();
       ++oDSVclusterInfoIter ){
#ifdef DEBUG
    std::cout << "inside loop over modules" << std::endl;
#endif
  
    // NO CHECK
    // map must be created, 
    // which allows to get the cluster-clusterinfo matching
    // without further loops on clusterinfos
    // ----------------------------------------------------
  
    // Extract ClusterInfos collection for given module
    oClusterInfos = oDSVclusterInfoIter->data;
    oDigis        = oDSVDigis->operator[]( oDSVclusterInfoIter->id).data;

    // Loop over ClusterInfos collection
    // ---------------------------------
    std::vector<SiStripClusterInfo>::iterator oIter;
    for( oIter= oClusterInfos.begin(); oIter != oClusterInfos.end();  ++oIter ){
      
      // Initialization
      // --------------
      module           = -99;
      type             = -99;
      monostereo       = -99;
      layer            = -99;
      bwfw             = -99;
      extint           = -99;
      string           = -99;
      wheel            = -99;
      rod              = -99;
      size             = -99;
      clusterpos       = -99.;
      clusterchg       = -99.;
      clusterchgl      = -99.;
      clusterchgr      = -99.;
      clusternoise     = -99.;
      clustermaxchg    = -99.;
      clustereta       = -99.;
      clustercrosstalk = -99.;
      

      // Create the AnalyzedCluster object
      AnalyzedCluster analyzedCluster;
      
      StripSubdetector oStripSubdet(oIter->geographicalId());
      GetSubDetInfo(oStripSubdet);
      
      // ClusterInfo was not processed yet
      // ---------------------------------
      size             = (int)oIter->width(); // cluster width
      clusterpos       =      oIter->position();
      clusterchg       =      oIter->charge();
      clusterchgl      =      oIter->chargeL();
      clusterchgr      =      oIter->chargeR();
      clusternoise     =      oIter->noise();
      clustermaxchg    =      oIter->maxCharge();
      clustereta       = getClusterEta(oIter->stripAmplitudes(), 
				       oIter->firstStrip(),
				       oDigis);
      clustercrosstalk = getClusterCrossTalk(oIter->stripAmplitudes(),
					     oIter->firstStrip(),
					     oDigis);
      
      geoId            = oIter->geographicalId();
      firstStrip       = oIter->firstStrip();
      clusterstripnoises = oIter->stripNoises();

      uint32_t detid = oIter->geographicalId();
      edm::DetSetVector<SiStripCluster>::const_iterator DSViter = oDSVCluster->find(detid);
      edm::DetSet<SiStripCluster>::const_iterator ClusIter = DSViter->data.begin();
      for(; ClusIter!=DSViter->data.end(); ++ClusIter){
	if (ClusIter->firstStrip() == oIter->firstStrip()){
	  clusterbarycenter = ClusIter->barycenter();
	  clusterseednoise = oIter->stripNoises()[( (int) ClusIter->barycenter() - 
						    ClusIter->firstStrip()         )];




	  // This is taken from "RecoLocalTracker/SiStripRecHitConverter/src/StripCPE.cc" in CMSSW_1_3_1
	  // -----------------------
	  //
	  // get the det from the geometry
	  //
	  DetId detId(geoId);
	  const GeomDetUnit *  det = tracker->idToDetUnit(detId);
	  LocalPoint position;
	  //	  LocalError eresult;
	  //	  LocalVector drift=LocalVector(0,0,1);
	  const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)(det);
	  //  DetId detId(det.geographicalId());
	  const StripTopology &topol=(StripTopology&)stripdet->topology();
	  position = topol.localPosition(clusterbarycenter);
	  //	  eresult = topol.localError(cl.barycenter(),1/12.);
	  //  drift = driftDirection(stripdet);
	  //  float thickness=stripdet->specificSurface().bounds().thickness();
	  //  drift*=thickness;
	  LocalPoint  result=LocalPoint(position.x(),position.y(),0);
	  // -------------------------------------

// 	  // Cluster position
// 	  LclPos_X = hit->localPosition().x();
// 	  LclPos_Y = hit->localPosition().y();
// 	  LclPos_Z = hit->localPosition().z();
	    
// 	  GlobalPoint oRecHitGlobalPos = 
// 	    tracker->idToDet(hit->geographicalId())->toGlobal(hit->localPosition());
// 	  GlbPos_X = oRecHitGlobalPos.x();
// 	  GlbPos_Y = oRecHitGlobalPos.y();
// 	  GlbPos_Z = oRecHitGlobalPos.z();

	  // Cluster position
	  LclPos_X = result.x();
	  LclPos_Y = result.y();
	  LclPos_Z = result.z();
	    
	  GlobalPoint oRecHitGlobalPos = 
	    tracker->idToDet(detId)->toGlobal(result);
	  GlbPos_X = oRecHitGlobalPos.x();
	  GlbPos_Y = oRecHitGlobalPos.y();
	  GlbPos_Z = oRecHitGlobalPos.z();

	}
      }
      
      countAll++;
      
      // Fill AnalyzedCluster
      // --------------------
      analyzedCluster.run          = run;
      analyzedCluster.event        = event;
      analyzedCluster.module           = module;    
      analyzedCluster.type             = type;      
      analyzedCluster.monostereo       = monostereo;
      analyzedCluster.layer            = layer; 
      analyzedCluster.bwfw             = bwfw;
      analyzedCluster.rod              = rod;
      analyzedCluster.wheel            = wheel;
      analyzedCluster.extint           = extint;
      analyzedCluster.string           = string;
      analyzedCluster.size             = size;
      analyzedCluster.clusterpos       = clusterpos;
      analyzedCluster.clusterchg       = clusterchg;
      analyzedCluster.clusterchgl      = clusterchgl;
      analyzedCluster.clusterchgr      = clusterchgr;
      analyzedCluster.clusternoise     = clusternoise;
      analyzedCluster.clustermaxchg    = clustermaxchg;
      analyzedCluster.clustereta       = clustereta;
      analyzedCluster.clustercrosstalk = clustercrosstalk;
      analyzedCluster.geoId            = geoId;

      double thickness = moduleThickness( geoId );
      analyzedCluster.thickness        = thickness;

      // Cluster position
      analyzedCluster.LclPos_X = LclPos_X;
      analyzedCluster.LclPos_Y = LclPos_Y;
      analyzedCluster.LclPos_Z = LclPos_Z;
      analyzedCluster.GlbPos_X = GlbPos_X;
      analyzedCluster.GlbPos_Y = GlbPos_Y;
      analyzedCluster.GlbPos_Z = GlbPos_Z;

      analyzedCluster.firstStrip       = firstStrip;
      analyzedCluster.clusterstripnoises = clusterstripnoises;
      analyzedCluster.clusterbarycenter  = clusterbarycenter;
      analyzedCluster.clusterseednoise   = clusterseednoise;

      // Index of the cluster inside the collection. The vector is updated later
      // so the index is the size of the old vector
      analyzedCluster.clu_id.push_back( v_anaclu_ptr->size() );

      // Must also fill a map containing <pair<uint32_t, int>, 
      // pointer to analyzedCluster which was filled in the vector
      v_anaclu_ptr->push_back(analyzedCluster);

      // v_anaclu_ptr->size() - 1
      // is the current index of the cluster stored in the vector
      map_analyzedCluster.insert(make_pair(make_pair(oIter->geographicalId(), 
						     oIter->firstStrip()   ),
					   v_anaclu_ptr->size()-1)  );


    } // Loop over ClusterInfos collection [oIter]
#ifdef DEBUG
    std::cout << "end of loop over modules" << std::endl;
#endif
  } // Loop over DetSetVector ClusterInfo [oDSVclusterInfoIter]

  // Perform track study
  // -------------------
  if(numberoftracks>0){

    int nTrackNum = 0;
    // Loop on track collection   
    // ------------------------
#ifdef DEBUG
    std::cout << "Starting to loop on all the track collection" << std::endl;
#endif
    reco::TrackCollection::const_iterator trackIter;
    for( trackIter=trackCollection->begin(); trackIter!=trackCollection->end(); ++trackIter ){

      // Create the AnalyzedTrack object
      AnalyzedTrack analyzedTrack;

      ++nTrackNum;
      ++trackcounter;  // only count processed tracks (not tracks->size())

      momentum     = trackIter->p();
      pt           = trackIter->pt();
      charge       = trackIter->charge();
      eta          = trackIter->eta();
      phi          = trackIter->phi();
      hitspertrack = trackIter->recHitsSize();
      normchi2     = trackIter->normalizedChi2();
      chi2         = trackIter->chi2();
      ndof         = trackIter->ndof();
      d0           = trackIter->d0();
      vx           = trackIter->vx();
      vy           = trackIter->vy();
      vz           = trackIter->vz();
      outerPt      = trackIter->outerPt();

      // Set the id here, so that tk_id = 0 if not on track
      //                            and > 0 for track index
      // --------------------------------------------------
      tk_id        = nTrackNum;

      reco::TrackRef trackref=reco::TrackRef(trackCollection,nTrackNum-1);
#ifdef DEBUG
      std::cout << "Track pt= " << trackref->pt() << std::endl;
#endif
      reco::TrackInfoRef trackinforef = (*TItkassociatorCollection.product())[trackref];
	
      // The track seed:
      // ---------------
      const TrajectorySeed seed = trackinforef->seed();
#ifdef DEBUG
      std::cout << "Test trackinfores seeds = " << seed.nHits()                << std::endl;
      std::cout << " N hits on Track = "        << (*trackIter).recHitsSize() << std::endl;
#endif

      // Disentangle matched rechits and evaluate corresponding
      // local directions and angles
      // The method requires only the trackinfo.
      // ---------------------------------------
      std::vector<std::pair<const TrackingRecHit *,float> > hitangle;
      std::auto_ptr<std::vector<std::pair<const TrackingRecHit *,float> > > hitangleXZ;
      std::auto_ptr<std::vector<std::pair<const TrackingRecHit *,float> > > hitangleYZ;
      std::auto_ptr<std::vector<std::pair<const TrackingRecHit *,float> > > hit3dangle;
      std::auto_ptr<std::vector<std::pair<const TrackingRecHit *,LocalVector> > > hitLclVector;
      std::auto_ptr<std::vector<std::pair<const TrackingRecHit *,GlobalVector> > > hitGlbVector;
      hitangle   = Anglefinder->SeparateHits(trackinforef);
      hitangleXZ = Anglefinder->getXZHitAngle();
      hitangleYZ = Anglefinder->getYZHitAngle();
      hit3dangle = Anglefinder->getHit3DAngle();
      hitLclVector = Anglefinder->getLocalDir();
      hitGlbVector = Anglefinder->getGlobalDir();

      int nHitNum = 0;
      // Loop on hits belonging to this track
      // ------------------------------------
#ifdef DEBUG
      std::cout << "Starting to loop on all the track hits" << std::endl;
#endif
      std::vector<std::pair<const TrackingRecHit *,float> >::iterator recHitsIter;
      for( recHitsIter=hitangle.begin(); recHitsIter!=hitangle.end(); ++recHitsIter ){

//  	TrackLocalAngleTIF::HitAngleAssociation::reference hitsrefXZ = hitangleXZ[nHitNum];
//  	TrackLocalAngleTIF::HitAngleAssociation::reference hitsrefYZ = hitangleYZ[nHitNum];
//  	TrackLocalAngleTIF::HitAngleAssociation::reference hitsref3D = hit3dangle[nHitNum];
	  

	std::pair<const TrackingRecHit*, float> hitangleXZ( (*hitangleXZ)[nHitNum] );
	std::pair<const TrackingRecHit*, float> hitangleYZ( (*hitangleYZ)[nHitNum] );
	std::pair<const TrackingRecHit*, float> hitangle3D( (*hit3dangle)[nHitNum] );
	std::pair<const TrackingRecHit *, LocalVector> hitLclDir( (*hitLclVector)[nHitNum] );
	std::pair<const TrackingRecHit *, GlobalVector> hitGlbDir( (*hitGlbVector)[nHitNum] );


	// Taking the hits previously separated by TrackLocalAngleTIF
	// ----------------------------------------------------------
	const SiStripRecHit2D* hit = dynamic_cast< const SiStripRecHit2D* >(recHitsIter->first);
	if ( hit != NULL ){
	  //   //&&&&&&&&&&&&&&&& Global POS &&&&&&&&&&&&&&&&&&&&&&&&
	  //   const StripTopology &topol=(StripTopology&)_StripGeomDetUnit->topology();
	  //   MeasurementPoint mp(cluster->position(),rnd.Uniform(-0.5,0.5));
	  //   LocalPoint localPos = topol.localPosition(mp);
	  //   GlobalPoint globalPos=(_StripGeomDetUnit->surface()).toGlobal(localPos);
	  //   //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	  //--------------------------------------------------------------------------------

	  // Taking the SiStripClusters corresponding to the SiStripRecHits
	  // --------------------------------------------------------------
	  const edm::Ref<edm::DetSetVector<SiStripCluster>,
	    SiStripCluster,
	    edm::refhelper::FindForDetSetVector<SiStripCluster> > 
	    cluster = hit->cluster();
	    
#ifdef DEBUG
	  std::cout << "cluster->geographicalId() = "
		    << cluster->geographicalId() << std::endl;
#endif
	    
	  oClusterInfos = oDSVClusterInfos->operator[](cluster->geographicalId()).data;
	    
	  // Change here
	  // Use the map to check if there is a clusterinfo matching given cluster.
	  // In the case fill the additional infos.
	  clustermap::iterator analyzedClusterIter;
	  analyzedClusterIter = 
	    map_analyzedCluster.find(make_pair(cluster->geographicalId(),
					       cluster->firstStrip()     ));

	  if (analyzedClusterIter !=
	      map_analyzedCluster.end()) {

	    // Match found: ClusterInfo matched given cluster
	    ++nHitNum;
	    clusterTKcounter++;

	    float angle = recHitsIter->second;
	    float angleXZ = hitangleXZ.second;
	    float angleYZ = hitangleYZ.second;
	    float angle3D = hitangle3D.second;

	    float LclDir_X = hitLclDir.second.x();
	    float LclDir_Y = hitLclDir.second.y();
	    float LclDir_Z = hitLclDir.second.z();

	    float GlbDir_X = hitGlbDir.second.x();
	    float GlbDir_Y = hitGlbDir.second.y();
	    float GlbDir_Z = hitGlbDir.second.z();

	    // Local Magnetic Field
	    // --------------------
	    //   //&&&&&&&&&&&&&&&& Global POS &&&&&&&&&&&&&&&&&&&&&&&&
	    //   const StripTopology &topol=(StripTopology&)_StripGeomDetUnit->topology();
	    //   MeasurementPoint mp(cluster->position(),rnd.Uniform(-0.5,0.5));
	    //   LocalPoint localPos = topol.localPosition(mp);
	    //   GlobalPoint globalPos=(_StripGeomDetUnit->surface()).toGlobal(localPos);
	    //   //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	    //--------------------------------------------------------------------------------
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
	      StripSubdetector oStripSubdet(cluster->geographicalId());

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
		  LogDebug("AnaObjProducer::analyze") << "Error: TIB|TOB YBprojection = 0";
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
	      
	      
	    // Write all the additional infos on angles and ...
	    
	    int anaCluVecIndex = analyzedClusterIter->second;
	    
	    AnalyzedCluster* tmpAnaClu = &((*v_anaclu_ptr)[anaCluVecIndex]);

// 	    tmpAnaClu->LclPos_X.insert( make_pair( tk_id, LclPos_X ) );
// 	    tmpAnaClu->LclPos_Y.insert( make_pair( tk_id, LclPos_Y ) );
// 	    tmpAnaClu->LclPos_Z.insert( make_pair( tk_id, LclPos_Z ) );
// 	    tmpAnaClu->GlbPos_X.insert( make_pair( tk_id, GlbPos_X ) );
// 	    tmpAnaClu->GlbPos_Y.insert( make_pair( tk_id, GlbPos_Y ) );
// 	    tmpAnaClu->GlbPos_Z.insert( make_pair( tk_id, GlbPos_Z ) );

	    tmpAnaClu->LclDir_X.insert( make_pair( tk_id, LclDir_X ) );
	    tmpAnaClu->LclDir_Y.insert( make_pair( tk_id, LclDir_Y ) );
	    tmpAnaClu->LclDir_Z.insert( make_pair( tk_id, LclDir_Z ) );
	    tmpAnaClu->GlbDir_X.insert( make_pair( tk_id, GlbDir_X ) );
	    tmpAnaClu->GlbDir_Y.insert( make_pair( tk_id, GlbDir_Y ) );
	    tmpAnaClu->GlbDir_Z.insert( make_pair( tk_id, GlbDir_Z ) );

	    tmpAnaClu->tk_id.push_back( tk_id );

	    tmpAnaClu->tk_phi.insert( make_pair( tk_id, tk_phi) );
	    tmpAnaClu->tk_theta.insert( make_pair( tk_id, tk_theta) );

	    tmpAnaClu->sign.insert( make_pair( tk_id, sign) );
	    tmpAnaClu->angle.insert( make_pair( tk_id, angle) );
	    tmpAnaClu->angleXZ.insert( make_pair( tk_id, angleXZ) );
	    tmpAnaClu->angleYZ.insert( make_pair( tk_id, angleYZ) );
	    tmpAnaClu->angle3D.insert( make_pair( tk_id, angle3D) );

	    tmpAnaClu->stereocorrection.insert( make_pair( tk_id, stereocorrection ) );
	    tmpAnaClu->localmagfield.insert( make_pair( tk_id, localmagfield ) );

	    // For now use the pointer
	    //	    analyzedTrack.vecClusterOwned.push_back(tmpAnaClu);
	    // Work in progress to insert the Ref
	    //	    edm::Ref< std::vector<AnalyzedCluster>,AnalyzedCluster > tmpRefAnaClu = 
	    //	      edm::Ref< std::vector<AnalyzedCluster>,AnalyzedCluster > ((*v_anaclu_ptr),anaCluVecIndex);
	    //	    analyzedTrack.vecRefClusterOwned.push_back(tmpRefAnaClu);
      
	    countOn++;

	    // Add the cross references
	    // ------------------------
            anatk_id = tk_id-1;
	    tmpAnaClu->TrackRef( AnalyzedTrackRef( anatk_refprod, anatk_id ) );
            anaclu_id = anaCluVecIndex;
	    analyzedTrack.ClusterRef( AnalyzedClusterRef( anaclu_refprod, anaclu_id ) );

            // Insert also the index of the cluster in the collection
	    analyzedTrack.clu_id.push_back( anaCluVecIndex );

	  } // analyzedClusterIter != map_analyzedCluster.end()
	  else {
	    // No matching, no clusterinfo found for given cluster
#ifdef DEBUG
	    std::cout << "Error, no clusterinfo was found for cluster with:" << std::endl;
	    std::cout << "geographicalId = " << cluster->geographicalId()    << std::endl;
	    std::cout << "firstStrip = "     << cluster->firstStrip()        << std::endl;
#endif
	  }
	} // hit != NULL

	nTrackClusters += nHitNum;
	
      } // Loop on hits belonging to this track [recHitsIter]
      
      // Fill AnalyzedTrack
      // ------------------
      analyzedTrack.momentum     = momentum;
      analyzedTrack.pt           = pt;
      analyzedTrack.charge       = charge;
      analyzedTrack.eta          = eta;
      analyzedTrack.phi          = phi;
      analyzedTrack.hitspertrack = hitspertrack;
      analyzedTrack.normchi2     = normchi2;
      analyzedTrack.chi2         = chi2;
      analyzedTrack.ndof         = ndof;
      analyzedTrack.d0           = d0;
      analyzedTrack.vx           = vx;
      analyzedTrack.vy           = vy;
      analyzedTrack.vz           = vz;
      analyzedTrack.outerPt      = outerPt;
      analyzedTrack.tk_id        = tk_id;


      v_anatk_ptr->push_back(analyzedTrack);

    } // Loop on track collection [trackIter]
  } // numberoftracks > 0

  countOff = countAll - countOn;



  // Fill cluster tree
  // -----------------
  int numAnaClu = v_anaclu_ptr->size();
#ifdef DEBUG
  std::cout << "numAnaClu= " << numAnaClu << std::endl;
  std::cout << " countAll= " << countAll << std::endl;
  if(numAnaClu!=countAll) std::cout << "numAnaClu= " << numAnaClu 
				    << " countAll= " << countAll << std::endl;
#endif

  for(int AnaCluIter=0; AnaCluIter!=numAnaClu; ++AnaCluIter){
    
    AnalyzedCluster tmpAnaClu = (*v_anaclu_ptr)[AnaCluIter];

    run              = tmpAnaClu.run;
    event            = tmpAnaClu.event;
    module           = tmpAnaClu.module;
    type             = tmpAnaClu.type;
    monostereo       = tmpAnaClu.monostereo;
    layer            = tmpAnaClu.layer;
    bwfw             = tmpAnaClu.bwfw;
    rod              = tmpAnaClu.rod;
    wheel            = tmpAnaClu.wheel;
    extint           = tmpAnaClu.extint;
    string           = tmpAnaClu.string;
    size             = tmpAnaClu.size;
    clusterpos       = tmpAnaClu.clusterpos;
    clusterchg       = tmpAnaClu.clusterchg;
    clusterchgl      = tmpAnaClu.clusterchgl;
    clusterchgr      = tmpAnaClu.clusterchgr;
    clusternoise     = tmpAnaClu.clusternoise;
    clustermaxchg    = tmpAnaClu.clustermaxchg;
    clustereta       = tmpAnaClu.clustereta;
    clustercrosstalk = tmpAnaClu.clustercrosstalk;
    geoId            = tmpAnaClu.geoId;
    firstStrip       = tmpAnaClu.firstStrip;
    clusterseednoise  = tmpAnaClu.clusterseednoise;
    clusterbarycenter = tmpAnaClu.clusterbarycenter;


    int numAssTrk = tmpAnaClu.tk_id.size();
    if(numAssTrk!=0){
      for(int AssTrkIter=0; AssTrkIter!=numAssTrk; ++AssTrkIter){

// 	dLclX = tmpAnaClu.dLclX[AssTrkIter];
// 	dLclY = tmpAnaClu.dLclY[AssTrkIter];
// 	dLclZ = tmpAnaClu.dLclZ[AssTrkIter];
// 	dGlbX = tmpAnaClu.dGlbX[AssTrkIter];
// 	dGlbY = tmpAnaClu.dGlbY[AssTrkIter];
// 	dGlbZ = tmpAnaClu.dGlbZ[AssTrkIter];
	
// 	tk_id    = tmpAnaClu.tk_id[AssTrkIter];
// 	tk_phi   = tmpAnaClu.tk_phi[AssTrkIter];
// 	tk_theta = tmpAnaClu.tk_theta[AssTrkIter];
	
// 	sign             = tmpAnaClu.sign[AssTrkIter];
// 	angle            = tmpAnaClu.angle[AssTrkIter];
// 	stereocorrection = tmpAnaClu.stereocorrection[AssTrkIter];
// 	localmagfield    = tmpAnaClu.localmagfield[AssTrkIter];
	
	clustercounter++;
      }
    }else{
      
//       dLclX = -99;
//       dLclY = -99;
//       dLclZ = -99;
//       dGlbX = -99;
//       dGlbY = -99;
//       dGlbZ = -99;
      
//       tk_id    = 0;
//       tk_phi   = -99.;
//       tk_theta = -99.;
      
//       clusterseednoise  = -99.;
//       clusterbarycenter = -99.;
      
//       sign             = -99;
//       angle            = -99.;
//       stereocorrection = -99.;
//       localmagfield    = -99.;
      
      clustercounter++;
      clusterNNTKcounter++;
    }
  }
  
  // Fill track tree
  // ---------------

  int numAnaTrk = v_anatk_ptr->size();
#ifdef DEBUG
  if(numAnaTrk!=numberoftracks) std::cout << "numAnaTrk= "       << numAnaTrk 
					  << " numberoftracks= " << numberoftracks << std::endl;
#endif

  for(int AnaTrkIter=0; AnaTrkIter!=numAnaTrk; ++AnaTrkIter){
    
    AnalyzedTrack tmpAnaTrk = (*v_anatk_ptr)[AnaTrkIter];

    momentum     = tmpAnaTrk.momentum;
    pt           = tmpAnaTrk.pt;
    charge       = tmpAnaTrk.charge;
    eta          = tmpAnaTrk.eta;
    phi          = tmpAnaTrk.phi;
    hitspertrack = tmpAnaTrk.hitspertrack;
    normchi2     = tmpAnaTrk.normchi2;
    ndof         = tmpAnaTrk.ndof;
    d0           = tmpAnaTrk.d0;
    vx           = tmpAnaTrk.vx;
    vy           = tmpAnaTrk.vy;
    vz           = tmpAnaTrk.vz;
    outerPt      = tmpAnaTrk.outerPt;
  }

// Fill the event with the analyzed objects collections
#ifdef DEBUG
  std::cout << "analyzedtrack_ before = " << analyzedtrack_ << std::endl;
  std::cout << "analyzedcluster_ before = " << analyzedcluster_ << std::endl;
#endif

  e.put(v_anatk_ptr, analyzedtrack_);
  e.put(v_anaclu_ptr, analyzedcluster_);

} // end produce


// EndJob
// ------
void AnaObjProducer::endJob(){

#ifdef DEBUG
  if(clusterNNTKcounter != (clustercounter - clusterTKcounter)) 
    std::cout << "clusterNNTKcouter= " << clusterNNTKcounter 
	      << " (clustercounter - clusterTKcounter)= " << (clustercounter - clusterTKcounter) << std::endl;
  std::cout << "endJob"                                                                << std::endl;
#endif

  std::cout << ">>> TOTAL EVENT = "                              << eventcounter       << std::endl;
  std::cout << ">>> NUMBER OF TRACKS = "                         << trackcounter       << std::endl;
  std::cout << ">>> NUMBER OF CLUSTER = "                        << clustercounter     << std::endl;
  std::cout << ">>> NUMBER OF CLUSTER BELONGING TO TRACKS = "    << clusterTKcounter   << std::endl;
  std::cout << ">>> NUMBER OF CLUSTER NOT BELONGING TO TRACKS = "<< clusterNNTKcounter << std::endl;

}
//------------------------------------------------------------------------
void AnaObjProducer::GetSubDetInfo(StripSubdetector oStripSubdet){

  std::string cSubDet;
  module     = oStripSubdet.rawId();
  type       = oStripSubdet.subdetId();
  monostereo = oStripSubdet.stereo();

#ifdef DEBUG
  std::cout << "type = " << type << std::endl;
#endif

  switch(type) {
  case StripSubdetector::TIB:
    {
      if(conf_.getParameter<bool>("TIB_ON")){
	TIBDetId oTIBDetId( oStripSubdet);
	layer   = oTIBDetId.layer();
	bwfw    = oTIBDetId.string()[0];
	extint  = oTIBDetId.string()[1];
	string  = oTIBDetId.string()[2];

	nTotClustersTIB++;
	cSubDet = "TIB";
      }
      break;
    }
  case StripSubdetector::TID:
    {
      if(conf_.getParameter<bool>("TID_ON")){
	TIDDetId oTIDDetId( oStripSubdet);
	wheel   = oTIDDetId.wheel();
	bwfw    = oTIDDetId.module()[0];

	nTotClustersTID++;
	cSubDet = "TID";
      }
      break;
    }
  case StripSubdetector::TOB:
    {
      if(conf_.getParameter<bool>("TOB_ON")) {
	TOBDetId oTOBDetId( oStripSubdet);
	layer   = oTOBDetId.layer();
	bwfw    = oTOBDetId.rod()[0];
	rod     = oTOBDetId.rod()[1];

	nTotClustersTOB++;
	cSubDet = "TOB";
      }
      break;
    }
  case StripSubdetector::TEC:
    {
      if(conf_.getParameter<bool>("TEC_ON")) {
	TECDetId oTECDetId( oStripSubdet);
	wheel   = oTECDetId.wheel();
	bwfw    = oTECDetId.petal()[0];

	nTotClustersTEC++;
	cSubDet = "TEC";
      }
      break;
    }
  }
}
//-----------------------------------------------------------------------
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
double AnaObjProducer::getClusterEta( const std::vector<uint16_t> &roSTRIP_AMPLITUDES,
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
double AnaObjProducer::getClusterCrossTalk( const std::vector<uint16_t> 
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
AnaObjProducer::calculateClusterCrossTalk( const double &rdADC_STRIPL,
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
  AnaObjProducer::getClusterCrossTalk( const std::vector<uint16_t> 
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

// double AnaObjProducer::moduleThickness(const TrackingRecHit* hit)
double AnaObjProducer::moduleThickness(const uint32_t detid)
{
  double t=0.;

  const GeomDetUnit* it = tracker->idToDetUnit(DetId(detid));
  //FIXME throw exception (taken from RecoLocalTracker/SiStripClusterizer/src/SiStripNoiseService.cc)
  if (dynamic_cast<const StripGeomDetUnit*>(it)==0 && dynamic_cast<const PixelGeomDetUnit*>(it)==0) {
    cout << "\t\t this detID doesn't seem to belong to the Tracker" << endl;
  }else{
    t = it->surface().bounds().thickness();
  }
  //cout << "\t\t thickness = " << t << endl;

  return t;
}
