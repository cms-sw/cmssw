// Package:    TrackerMonitorTrack
// Class:      SiPixelSiPixelMonitorTrackResiduals
// 
// class SiPixelMonitorTrackResiduals SiPixelMonitorTrackResiduals.cc 
//       DQM/TrackerMonitorTrack/src/SiPixelMonitorTrackResiduals.cc
//
// Description:    <one line class summary>
// Implementation: <Notes on implementation>
//
// Original Author: Shan-Huei Chuang
//         Created: Fri Mar 23 18:41:42 CET 2007
// $Id: SiPixelMonitorTrackResiduals.cc, v0.0 2007/03/23 18:41:42 schuang Exp $


#include "DQM/SiPixelMonitorTrack/interface/SiPixelMonitorTrackResiduals.h"


using namespace std;
using namespace edm;


SiPixelMonitorTrackResiduals::SiPixelMonitorTrackResiduals(const edm::ParameterSet& iConfig) {
  dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();
  conf_ = iConfig;
}


SiPixelMonitorTrackResiduals::~SiPixelMonitorTrackResiduals() {
}


void SiPixelMonitorTrackResiduals::beginJob(edm::EventSetup const& iSetup) {
  std::cout << " *** SiPixelMonitorTrackResiduals " << std::endl;

  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get(pDD);

  std::cout <<" *** Geometry node for TrackerGeom is " << &(*pDD) << std::endl;
  std::cout <<" *** " << pDD->dets().size() <<" detectors; "
                      << pDD->detTypes().size() <<" types" << std::endl;
  
  for (TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it!=pDD->dets().end(); it++) {
    if (dynamic_cast<PixelGeomDetUnit*>((*it))!=0) {
      DetId detId = (*it)->geographicalId();

      // const GeomDetUnit* geoUnit = pDD->idToDetUnit(detId);
      // const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(geoUnit);
      // int nrows = (pixDet->specificTopology()).nrows();
      // int ncols = (pixDet->specificTopology()).ncolumns();
      
      if (detId.subdetId()==static_cast<int>(PixelSubdetector::PixelBarrel)) {
	uint32_t id = detId();
	SiPixelTrackResModule* pippo = new SiPixelTrackResModule(id);
	thePixelStructure.insert(pair<uint32_t, SiPixelTrackResModule*> (id, pippo));
      }	
      else if (detId.subdetId()==static_cast<int>(PixelSubdetector::PixelEndcap)) {
	uint32_t id = detId();
	SiPixelTrackResModule* pippo = new SiPixelTrackResModule(id);
	thePixelStructure.insert(pair<uint32_t, SiPixelTrackResModule*> (id, pippo));
      }
    }
  }
  std::cout << " *** size of thePixelStructure is " << thePixelStructure.size() << std::endl; 
  dbe_->setVerbose(0);
  std::string rootDir = "SiPixel";

  dbe_->setCurrentFolder(rootDir.c_str()); 
  char hkey[80]; 
  for (int sub=0; sub<3; sub++) {
    sprintf(hkey,"hitResidual-x_subdet%i",sub); 
    meSubpixelHitResidualX[sub] = dbe_->book1D(hkey,"Hit Residual in X",1000,-5.,5.);
    
    sprintf(hkey,"hitResidual-y_subdet%i",sub); 
    meSubpixelHitResidualY[sub] = dbe_->book1D(hkey,"Hit Residual in Y",1000,-5.,5.);  
  }
  std::map<uint32_t, SiPixelTrackResModule*>::iterator struct_iter;
  for (struct_iter = thePixelStructure.begin(); struct_iter!=thePixelStructure.end(); struct_iter++) {
    if (DetId::DetId((*struct_iter).first).subdetId()==static_cast<int>(PixelSubdetector::PixelBarrel)) {
      PixelBarrelName::Shell 
          DBshell  = PixelBarrelName::PixelBarrelName(DetId::DetId((*struct_iter).first)).shell();
      int DBlayer  = PixelBarrelName::PixelBarrelName(DetId::DetId((*struct_iter).first)).layerName();
      int DBladder = PixelBarrelName::PixelBarrelName(DetId::DetId((*struct_iter).first)).ladderName();
      int DBmodule = PixelBarrelName::PixelBarrelName(DetId::DetId((*struct_iter).first)).moduleName();
      
      std::string ssubdet = "Barrel"; 
      char slayer[80];  sprintf(slayer, "Layer_%i",   DBlayer );
      char sladder[80]; sprintf(sladder,"Ladder_%02i",DBladder);
      char smodule[80]; sprintf(smodule,"Module_%i",  DBmodule);

      std::ostringstream sfolder;
      sfolder << rootDir << "/" << ssubdet << "/Shell_" << DBshell << "/" << slayer << "/" << sladder;
      if (PixelBarrelName::PixelBarrelName(DetId::DetId((*struct_iter).first)).isHalfModule()) sfolder << "H"; 
      else sfolder << "F";
      sfolder << "/" << smodule;

      dbe_->setCurrentFolder(sfolder.str().c_str());
      (*struct_iter).second->book();
    } 
    else if (DetId::DetId((*struct_iter).first).subdetId()==static_cast<int>(PixelSubdetector::PixelEndcap)) {
      std::string ssubdet = "Endcap";
      PixelEndcapName::HalfCylinder 
          side   = PixelEndcapName::PixelEndcapName(DetId::DetId((*struct_iter).first)).halfCylinder();
      int disk   = PixelEndcapName::PixelEndcapName(DetId::DetId((*struct_iter).first)).diskName();
      int blade  = PixelEndcapName::PixelEndcapName(DetId::DetId((*struct_iter).first)).bladeName();
      int panel  = PixelEndcapName::PixelEndcapName(DetId::DetId((*struct_iter).first)).pannelName();
      int module = PixelEndcapName::PixelEndcapName(DetId::DetId((*struct_iter).first)).plaquetteName();

      char sdisk[80];   sprintf(sdisk,  "Disk_%i",   disk);
      char sblade[80];  sprintf(sblade, "Blade_%02i",blade);
      char spanel[80];  sprintf(spanel, "Panel_%i",  panel);
      char smodule[80]; sprintf(smodule,"Module_%i", module);

      std::ostringstream sfolder;
      sfolder << rootDir << "/" << ssubdet << "/HalfCylinder_" << side << "/" << sdisk << "/" << sblade << "/" << spanel << "/" << smodule;
      dbe_->setCurrentFolder(sfolder.str().c_str());
      (*struct_iter).second->book();
    }
  } 
}


void SiPixelMonitorTrackResiduals::endJob(void) {
  dbe_->showDirStructure();
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if (outputMEsInRootFile) dbe_->save(outputFileName);
}


void SiPixelMonitorTrackResiduals::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::string TrackCandidateLabel = conf_.getParameter<std::string>("TrackCandidateLabel");
  std::string TrackCandidateProducer = conf_.getParameter<std::string>("TrackCandidateProducer");

  ESHandle<TrackerGeometry> theRG;
  iSetup.get<TrackerDigiGeometryRecord>().get(theRG);
  
  ESHandle<MagneticField> theRMF;
  iSetup.get<IdealMagneticFieldRecord>().get(theRMF);
  
  ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  iSetup.get<TransientRecHitRecord>().get("WithTrackAngle",theBuilder);
  
  ESHandle<TrajectoryFitter> theRFitter;
  iSetup.get<TrackingComponentsRecord>().get("KFFittingSmoother",theRFitter);
 
  const TransientTrackingRecHitBuilder* builder = theBuilder.product();
  const TrackerGeometry* theG = theRG.product();
  const MagneticField* theMF = theRMF.product();
  const TrajectoryFitter* theFitter = theRFitter.product();

  Handle<TrackCandidateCollection> trackCandidateCollection;
  iEvent.getByLabel(TrackCandidateProducer, TrackCandidateLabel, trackCandidateCollection);

  for (TrackCandidateCollection::const_iterator track = trackCandidateCollection->begin(); 
       track!=trackCandidateCollection->end(); ++track) {
    const TrackCandidate* theTC = &(*track);
    PTrajectoryStateOnDet state = theTC->trajectoryStateOnDet();
    const TrackCandidate::range& recHitVec = theTC->recHits();
    const TrajectorySeed& seed = theTC->seed();
    std::cout <<" with "<< (int)(recHitVec.second - recHitVec.first) <<" hits "<< std::endl;

    // convert PTrajectoryStateOnDet to TrajectoryStateOnSurface
    TrajectoryStateTransform transformer;

    DetId detId(state.detId());
    TrajectoryStateOnSurface theTSOS = transformer.transientState(state, &(theG->idToDet(detId)->surface()), theMF);

    Trajectory::RecHitContainer hits;
    
    TrackingRecHitCollection::const_iterator hit;
    for (hit = recHitVec.first; hit!=recHitVec.second; ++hit) hits.push_back(builder->build(&(*hit)));

    std::vector<Trajectory> trajVec = theFitter->fit(seed, hits, theTSOS);
    std::cout <<" fitted candidate with "<< trajVec.size() <<" tracks "<< std::endl;

    if (trajVec.size()!=0) {
      const Trajectory& theTraj = trajVec.front();

      Trajectory::DataContainer fits = theTraj.measurements();
      for (Trajectory::DataContainer::iterator fit = fits.begin(); fit!=fits.end(); fit++) {
        const TrajectoryMeasurement tm = *fit;
        TrajectoryStateOnSurface theCombinedPredictedState = TrajectoryStateCombiner().combine(tm.forwardPredictedState(),
	                                                                                       tm.backwardPredictedState());
        TransientTrackingRecHit::ConstRecHitPointer hit = tm.recHit();
        const GeomDet* det = hit->det();
                  
        // check that the detector module belongs to the Silicon Pixel detector
        if (det->components().empty() && (det->subDetector()==GeomDetEnumerators::PixelBarrel ||
                			  det->subDetector()==GeomDetEnumerators::PixelEndcap)) {
          const GeomDetUnit* du = dynamic_cast<const GeomDetUnit*>(det);
          const Topology* theTopol = &(du->topology());

          // calculate hit residuals in the measurement frame 
          MeasurementPoint theMeasHitPos = theTopol->measurementPosition(hit->localPosition());
          MeasurementPoint theMeasStatePos = theTopol->measurementPosition(theCombinedPredictedState.localPosition());
          Measurement2DVector hitResidual = theMeasHitPos - theMeasStatePos;
	  
	  // fill histograms by module id and then by subdetector
	  DetId hit_detId = hit->geographicalId(); 
	  int IntRawDetID = hit_detId.rawId();           
          std::map<uint32_t, SiPixelTrackResModule*>::iterator struct_iter = thePixelStructure.find(IntRawDetID);
	  if (struct_iter!=thePixelStructure.end()) (*struct_iter).second->fill(hitResidual);
	  
	  if (det->subDetector()==GeomDetEnumerators::PixelEndcap) {
            PXFDetId pxf(hit_detId);
            meSubpixelHitResidualX[pxf.side()]->Fill(hitResidual.x());
	    meSubpixelHitResidualY[pxf.side()]->Fill(hitResidual.y());	  
	  }
	  else {
            meSubpixelHitResidualX[0]->Fill(hitResidual.x());
	    meSubpixelHitResidualY[0]->Fill(hitResidual.y());	  
	  }
     	}
      }
    }
  } 
}


// define this as a plug-in
// DEFINE_FWK_MODULE(SiPixelMonitorTrackResiduals) 
