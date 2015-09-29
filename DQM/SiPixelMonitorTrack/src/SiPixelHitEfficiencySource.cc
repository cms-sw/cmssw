// Package:    SiPixelMonitorTrack
// Class:      SiPixelHitEfficiencySource
// 
// class SiPixelHitEfficiencyModule SiPixelHitEfficiencyModule.cc 
//       DQM/SiPixelMonitorTrack/src/SiPixelHitEfficiencyModule.cc
//
// Description: SiPixel hit efficiency data quality monitoring modules
// Implementation: prototype -> improved -> never final - end of the 1st step 
//
// Original Authors: Romain Rougny & Luca Mucibello
//         Created: Mar Nov 10 13:29:00 CET 2009


#include <iostream>
#include <map>
#include <math.h>
#include <string>
#include <vector>
#include <utility>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
//#include "RecoLocalTracker/SiPixelRecHits/plugins/PixelCPEGenericESProducer.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
#include "DQM/SiPixelMonitorTrack/interface/SiPixelHitEfficiencySource.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

using namespace std;
using namespace edm;


SiPixelHitEfficiencySource::SiPixelHitEfficiencySource(const edm::ParameterSet& pSet) :
  pSet_(pSet),
  modOn( pSet.getUntrackedParameter<bool>("modOn",true) ),
  ladOn( pSet.getUntrackedParameter<bool>("ladOn",false) ), 
  layOn( pSet.getUntrackedParameter<bool>("layOn",false) ), 
  phiOn( pSet.getUntrackedParameter<bool>("phiOn",false) ), 
  ringOn( pSet.getUntrackedParameter<bool>("ringOn",false) ), 
  bladeOn( pSet.getUntrackedParameter<bool>("bladeOn",false) ), 
  diskOn( pSet.getUntrackedParameter<bool>("diskOn",false) ), 
  isUpgrade( pSet.getUntrackedParameter<bool>("isUpgrade",false) )
  //updateEfficiencies( pSet.getUntrackedParameter<bool>("updateEfficiencies",false) )
{ 
   pSet_ = pSet; 
   debug_ = pSet_.getUntrackedParameter<bool>("debug", false); 
   applyEdgeCut_ = pSet_.getUntrackedParameter<bool>("applyEdgeCut");
   nSigma_EdgeCut_ = pSet_.getUntrackedParameter<double>("nSigma_EdgeCut");
   vertexCollectionToken_ = consumes<reco::VertexCollection>(std::string("offlinePrimaryVertices"));
   tracksrc_ = consumes<TrajTrackAssociationCollection>(pSet_.getParameter<edm::InputTag>("trajectoryInput"));
   clusterCollectionToken_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(std::string("siPixelClusters"));

   measurementTrackerEventToken_ = consumes<MeasurementTrackerEvent>(std::string("MeasurementTrackerEvent"));

   firstRun = true;
   
   LogInfo("PixelDQM") << "SiPixelHitEfficiencySource constructor" << endl;
   LogInfo ("PixelDQM") << "Mod/Lad/Lay/Phi " << modOn << "/" << ladOn << "/" 
			<< layOn << "/" << phiOn << std::endl;
   LogInfo ("PixelDQM") << "Blade/Disk/Ring" << bladeOn << "/" << diskOn << "/" 
			<< ringOn << std::endl;
}


SiPixelHitEfficiencySource::~SiPixelHitEfficiencySource() {
  LogInfo("PixelDQM") << "SiPixelHitEfficiencySource destructor" << endl;

  std::map<uint32_t,SiPixelHitEfficiencyModule*>::iterator struct_iter;
  for (struct_iter = theSiPixelStructure.begin() ; struct_iter != theSiPixelStructure.end() ; struct_iter++){
    delete struct_iter->second;
    struct_iter->second = 0;
  }
}

void SiPixelHitEfficiencySource::fillClusterProbability(int layer, int disk, bool plus, double probability){
 
  //barrel
  if (layer!=0){
    if (layer==1){
      if (plus)  meClusterProbabilityL1_Plus_->Fill(probability);
      else  meClusterProbabilityL1_Minus_->Fill(probability);
    }

    else if (layer==2){
      if (plus)  meClusterProbabilityL2_Plus_->Fill(probability);
      else  meClusterProbabilityL2_Minus_->Fill(probability);
    }

    else if (layer==3){
      if (plus)  meClusterProbabilityL3_Plus_->Fill(probability);
      else  meClusterProbabilityL3_Minus_->Fill(probability);
    }
  }
  //Endcap
  if (disk!=0){
    if (disk==1){
      if (plus)  meClusterProbabilityD1_Plus_->Fill(probability);
      else  meClusterProbabilityD1_Minus_->Fill(probability);
    }
    if (disk==2){
      if (plus)  meClusterProbabilityD2_Plus_->Fill(probability);
      else  meClusterProbabilityD2_Minus_->Fill(probability);
    }      
  }    
}





void SiPixelHitEfficiencySource::dqmBeginRun(const edm::Run& r, edm::EventSetup const& iSetup) {
  LogInfo("PixelDQM") << "SiPixelHitEfficiencySource beginRun()" << endl;
  
  if(firstRun){
    // retrieve TrackerGeometry for pixel dets
  
    nvalid=0;
    nmissing=0;
  
    firstRun = false;
  }

  edm::ESHandle<TrackerGeometry> TG;
  iSetup.get<TrackerDigiGeometryRecord>().get(TG);
  if (debug_) LogVerbatim("PixelDQM") << "TrackerGeometry "<< &(*TG) <<" size is "<< TG->dets().size() << endl;
 
  // build theSiPixelStructure with the pixel barrel and endcap dets from TrackerGeometry
  for (TrackerGeometry::DetContainer::const_iterator pxb = TG->detsPXB().begin();  
       pxb!=TG->detsPXB().end(); pxb++) {
    if (dynamic_cast<PixelGeomDetUnit const *>((*pxb))!=0) {
      SiPixelHitEfficiencyModule* module = new SiPixelHitEfficiencyModule((*pxb)->geographicalId().rawId());
      theSiPixelStructure.insert(pair<uint32_t, SiPixelHitEfficiencyModule*>((*pxb)->geographicalId().rawId(), module));
    }
  }
  for (TrackerGeometry::DetContainer::const_iterator pxf = TG->detsPXF().begin(); 
       pxf!=TG->detsPXF().end(); pxf++) {
    if (dynamic_cast<PixelGeomDetUnit const *>((*pxf))!=0) {
      SiPixelHitEfficiencyModule* module = new SiPixelHitEfficiencyModule((*pxf)->geographicalId().rawId());
      theSiPixelStructure.insert(pair<uint32_t, SiPixelHitEfficiencyModule*>((*pxf)->geographicalId().rawId(), module));
    }
  }
  LogInfo("PixelDQM") << "SiPixelStructure size is " << theSiPixelStructure.size() << endl;
  
}

void SiPixelHitEfficiencySource::bookHistograms(DQMStore::IBooker & iBooker, edm::Run const & iRun, edm::EventSetup const & iSetup){

  // book residual histograms in theSiPixelFolder - one (x,y) pair of histograms per det
  SiPixelFolderOrganizer theSiPixelFolder(false);
  for (std::map<uint32_t, SiPixelHitEfficiencyModule*>::iterator pxd = theSiPixelStructure.begin(); 
       pxd!=theSiPixelStructure.end(); pxd++) {

    if(modOn){
      if (theSiPixelFolder.setModuleFolder(iBooker,(*pxd).first,0,isUpgrade)) (*pxd).second->book(pSet_,iSetup,iBooker,0,isUpgrade);
      else throw cms::Exception("LogicError") << "SiPixelHitEfficiencySource Folder Creation Failed! "; 
    }
    if(ladOn){
      if (theSiPixelFolder.setModuleFolder(iBooker,(*pxd).first,1,isUpgrade)) (*pxd).second->book(pSet_,iSetup,iBooker,1,isUpgrade);
      else throw cms::Exception("LogicError") << "SiPixelHitEfficiencySource ladder Folder Creation Failed! "; 
    }
    if(layOn){
      if (theSiPixelFolder.setModuleFolder(iBooker,(*pxd).first,2,isUpgrade)) (*pxd).second->book(pSet_,iSetup,iBooker,2,isUpgrade);
      else throw cms::Exception("LogicError") << "SiPixelHitEfficiencySource layer Folder Creation Failed! "; 
    }
    if(phiOn){
      if (theSiPixelFolder.setModuleFolder(iBooker,(*pxd).first,3,isUpgrade)) (*pxd).second->book(pSet_,iSetup,iBooker,3,isUpgrade);
      else throw cms::Exception("LogicError") << "SiPixelHitEfficiencySource phi Folder Creation Failed! "; 
    }
    if(bladeOn){
      if (theSiPixelFolder.setModuleFolder(iBooker,(*pxd).first,4,isUpgrade)) (*pxd).second->book(pSet_,iSetup,iBooker,4,isUpgrade);
      else throw cms::Exception("LogicError") << "SiPixelHitEfficiencySource Blade Folder Creation Failed! "; 
    }
    if(diskOn){
      if (theSiPixelFolder.setModuleFolder(iBooker,(*pxd).first,5,isUpgrade)) (*pxd).second->book(pSet_,iSetup,iBooker,5,isUpgrade);
      else throw cms::Exception("LogicError") << "SiPixelHitEfficiencySource Disk Folder Creation Failed! "; 
    }
    if(ringOn){
      if (theSiPixelFolder.setModuleFolder(iBooker,(*pxd).first,6,isUpgrade)) (*pxd).second->book(pSet_,iSetup,iBooker,6,isUpgrade);
      else throw cms::Exception("LogicError") << "SiPixelHitEfficiencySource Ring Folder Creation Failed! "; 
    }
  }

  //book cluster probability histos for Barrel and Endcap
  iBooker.setCurrentFolder("Pixel/Barrel");

  meClusterProbabilityL1_Plus_  = iBooker.book1D("ClusterProbabilityXY_Layer1_Plus","ClusterProbabilityXY_Layer1_Plus",250,-5,0.1);
  meClusterProbabilityL1_Plus_->setAxisTitle("Log(ClusterProbability)",1);

  meClusterProbabilityL1_Minus_ = iBooker.book1D("ClusterProbabilityXY_Layer1_Minus","ClusterProbabilityXY_Layer1_Minus",250,-5,0.1);
  meClusterProbabilityL1_Minus_->setAxisTitle("Log(ClusterProbability)",1);

  meClusterProbabilityL2_Plus_  = iBooker.book1D("ClusterProbabilityXY_Layer2_Plus","ClusterProbabilityXY_Layer2_Plus",250,-5,0.1);
  meClusterProbabilityL2_Plus_ ->setAxisTitle("Log(ClusterProbability)",1);

  meClusterProbabilityL2_Minus_ = iBooker.book1D("ClusterProbabilityXY_Layer2_Minus","ClusterProbabilityXY_Layer2_Minus",250,-5,0.1);
  meClusterProbabilityL2_Minus_ ->setAxisTitle("Log(ClusterProbability)",1);

  meClusterProbabilityL3_Plus_  = iBooker.book1D("ClusterProbabilityXY_Layer3_Plus","ClusterProbabilityXY_Layer3_Plus",250,-5,0.1);
  meClusterProbabilityL3_Plus_->setAxisTitle("Log(ClusterProbability)",1);
 
  meClusterProbabilityL3_Minus_ = iBooker.book1D("ClusterProbabilityXY_Layer3_Minus","ClusterProbabilityXY_Layer3_Minus",250,-5,0.1);
  meClusterProbabilityL3_Minus_->setAxisTitle("Log(ClusterProbability)",1);

  iBooker.setCurrentFolder("Pixel/Endcap");

  meClusterProbabilityD1_Plus_  = iBooker.book1D("ClusterProbabilityXY_Disk1_Plus","ClusterProbabilityXY_Disk1_Plus",250,-5,0.1);
  meClusterProbabilityD1_Plus_ ->setAxisTitle("Log(ClusterProbability)",1);

  meClusterProbabilityD1_Minus_ = iBooker.book1D("ClusterProbabilityXY_Disk1_Minus","ClusterProbabilityXY_Disk1_Minus",250,-5,0.1);
  meClusterProbabilityD1_Minus_->setAxisTitle("Log(ClusterProbability)",1);

  meClusterProbabilityD2_Plus_  = iBooker.book1D("ClusterProbabilityXY_Disk2_Plus","ClusterProbabilityXY_Disk2_Plus",250,-5,0.1);
  meClusterProbabilityD2_Plus_ ->setAxisTitle("Log(ClusterProbability)",1);

  meClusterProbabilityD2_Minus_ = iBooker.book1D("ClusterProbabilityXY_Disk2_Minus","ClusterProbabilityXY_Disk2_Minus",250,-5,0.1);
  meClusterProbabilityD2_Minus_->setAxisTitle("Log(ClusterProbability)",1);

}

void SiPixelHitEfficiencySource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology *pTT = tTopoHandle.product();

  edm::Handle<reco::VertexCollection> vertexCollectionHandle;
  iEvent.getByToken( vertexCollectionToken_, vertexCollectionHandle );
  if(!vertexCollectionHandle.isValid()) return;
  nvtx_=0;
  vtxntrk_=-9999;
  vtxD0_=-9999.; vtxX_=-9999.; vtxY_=-9999.; vtxZ_=-9999.; vtxndof_=-9999.; vtxchi2_=-9999.;
  const reco::VertexCollection & vertices = *vertexCollectionHandle.product();
  reco::VertexCollection::const_iterator bestVtx=vertices.end();
  for(reco::VertexCollection::const_iterator it=vertices.begin(); it!=vertices.end(); ++it){
    if(!it->isValid()) continue;
    if(vtxntrk_==-9999 ||
       vtxntrk_<int(it->tracksSize()) ||
       (vtxntrk_==int(it->tracksSize()) && fabs(vtxZ_)>fabs(it->z()))){
      vtxntrk_=it->tracksSize();
      vtxD0_=it->position().rho();
      vtxX_=it->x();
      vtxY_=it->y();
      vtxZ_=it->z();
      vtxndof_=it->ndof();
      vtxchi2_=it->chi2();
      bestVtx=it;
    }
    if(fabs(it->z())<=20. && fabs(it->position().rho())<=2. && it->ndof()>4) nvtx_++;
  }
  if(nvtx_<1) return;
    
  //get the map
  edm::Handle<TrajTrackAssociationCollection> match;
  iEvent.getByToken( tracksrc_, match );
  const TrajTrackAssociationCollection ttac = *(match.product());

  if(debug_){
    std::cout << "+++ NEW EVENT +++"<< std::endl;
    std::cout << "Map entries \t : " << ttac.size() << std::endl;
  }

  std::set<SiPixelCluster> clusterSet;
  //  TrajectoryStateCombiner tsoscomb;
  //define variables for extrapolation
  int extrapolateFrom_ = 2;
  int extrapolateTo_ = 1;
  float maxlxmatch_=0.2;
  float maxlymatch_=0.2;
  bool keepOriginalMissingHit_=true;
  ESHandle<MeasurementTracker> measurementTrackerHandle;

  iSetup.get<CkfComponentsRecord>().get(measurementTrackerHandle);
  
  edm::ESHandle<Chi2MeasurementEstimatorBase> est;
  iSetup.get<TrackingComponentsRecord>().get("Chi2",est);
  edm::Handle<MeasurementTrackerEvent> measurementTrackerEventHandle;
  iEvent.getByToken(measurementTrackerEventToken_, measurementTrackerEventHandle);
  edm::ESHandle<Propagator> prop;
  iSetup.get<TrackingComponentsRecord>().get("PropagatorWithMaterial",prop);
  Propagator* thePropagator = prop.product()->clone();
  //determines direction of the propagator => inward
  if (extrapolateFrom_>=extrapolateTo_) {
    thePropagator->setPropagationDirection(oppositeToMomentum);
  }
  TrajectoryStateCombiner trajStateComb;
  bool debug_ = false;
  
  //Loop over track collection
  for(TrajTrackAssociationCollection::const_iterator it =  ttac.begin();it !=  ttac.end(); ++it){
    //define vector to save extrapolated tracks
    std::vector<TrajectoryMeasurement> expTrajMeasurements;

    const edm::Ref<std::vector<Trajectory> > traj_iterator = it->key;  
    // Trajectory Map, extract Trajectory for this track
    reco::TrackRef trackref = it->val;
    //tracks++;
    
    bool isBpixtrack = false, isFpixtrack = false;
    int nStripHits=0; int L1hits=0; int L2hits=0; int L3hits=0; int L4hits=0; int D1hits=0; int D2hits=0; int D3hits=0;
    std::vector<TrajectoryMeasurement> tmeasColl =traj_iterator->measurements();
    std::vector<TrajectoryMeasurement>::const_iterator tmeasIt;
    //loop on measurements to find out what kind of hits there are
    for(tmeasIt = tmeasColl.begin();tmeasIt!=tmeasColl.end();tmeasIt++){
      //if(! tmeasIt->updatedState().isValid()) continue; NOT NECESSARY (I HOPE)
      TransientTrackingRecHit::ConstRecHitPointer testhit = tmeasIt->recHit();
      if(testhit->geographicalId().det() != DetId::Tracker) continue; 
      uint testSubDetID = (testhit->geographicalId().subdetId()); 
      const DetId & hit_detId = testhit->geographicalId();
      int hit_layer = 0;
      int hit_ladder = 0;
      int hit_mod = 0;
      int hit_disk = 0;
      
      if(testSubDetID==PixelSubdetector::PixelBarrel){
        isBpixtrack = true;
	hit_layer = PixelBarrelName(hit_detId,pTT,isUpgrade).layerName();
	
	hit_ladder = PXBDetId(hit_detId).ladder();
	hit_mod = PXBDetId(hit_detId).module();

	if(hit_layer==1) L1hits++;
	if(hit_layer==2) L2hits++;
	if(hit_layer==3) L3hits++;
	if(hit_layer==4) L4hits++;
      }
      if(testSubDetID==PixelSubdetector::PixelEndcap){
        isFpixtrack = true;
        hit_disk = PixelEndcapName(hit_detId,pTT,isUpgrade).diskName();
        
	if(hit_disk==1) D1hits++;
	if(hit_disk==2) D2hits++;
        if(hit_disk==3) D3hits++;
      }
      if(testSubDetID==StripSubdetector::TIB) nStripHits++;
      if(testSubDetID==StripSubdetector::TOB) nStripHits++;
      if(testSubDetID==StripSubdetector::TID) nStripHits++;
      if(testSubDetID==StripSubdetector::TEC) nStripHits++;
      //check if last valid hit is in Layer 2 or Disk 1

      bool lastValidL2 = false;      
      if ((testSubDetID == PixelSubdetector::PixelBarrel && hit_layer == extrapolateFrom_) 
	  || (testSubDetID == PixelSubdetector::PixelEndcap && hit_disk == 1)) {
	if (testhit->isValid()) {
	  if (tmeasIt == tmeasColl.end()-1) {
	    lastValidL2=true;
	  } else {
	    tmeasIt++;
	    TransientTrackingRecHit::ConstRecHitPointer nextRecHit = tmeasIt->recHit();
	    uint nextSubDetID = (nextRecHit->geographicalId().subdetId()); 
	    int nextlayer = PixelBarrelName(nextRecHit->geographicalId()).layerName();
	    if (nextSubDetID == PixelSubdetector::PixelBarrel && nextlayer==extrapolateTo_ ) {
	      lastValidL2=true; //&& !nextRecHit->isValid()) lastValidL2=true;
	    }
	    tmeasIt--;
	  }
	}
      }//end check last valid layer
      if (lastValidL2) {
	std::vector< const BarrelDetLayer*> pxbLayers = measurementTrackerHandle->geometricSearchTracker()->pixelBarrelLayers();
	const DetLayer* pxb1 = pxbLayers[extrapolateTo_-1];
	const MeasurementEstimator* estimator = est.product();
	const LayerMeasurements* theLayerMeasurements =    new LayerMeasurements(*measurementTrackerHandle, *measurementTrackerEventHandle);  
	const TrajectoryStateOnSurface tsosPXB2 = tmeasIt->updatedState();
	expTrajMeasurements = theLayerMeasurements->measurements(*pxb1, tsosPXB2, *thePropagator, *estimator);
	delete theLayerMeasurements;
	if ( !expTrajMeasurements.empty()) {
	  for(uint p=0; p<expTrajMeasurements.size();p++){
	    TrajectoryMeasurement pxb1TM(expTrajMeasurements[p]);
	    auto pxb1Hit = pxb1TM.recHit();
	    //remove hits with rawID == 0
	    if(pxb1Hit->geographicalId().rawId()==0){
	      expTrajMeasurements.erase(expTrajMeasurements.begin()+p);
	      continue;
	    }
	  }
	}
	//
      }
      //check if extrapolated hit to layer 1 one matches the original hit
      TrajectoryStateOnSurface chkPredTrajState=trajStateComb(tmeasIt->forwardPredictedState(), tmeasIt->backwardPredictedState());
      float chkx=chkPredTrajState.globalPosition().x();
      float chky=chkPredTrajState.globalPosition().y();
      float chkz=chkPredTrajState.globalPosition().z();
      LocalPoint chklp=chkPredTrajState.localPosition();
      if (testSubDetID == PixelSubdetector::PixelBarrel && hit_layer == extrapolateTo_) {
	// Here we will drop the extrapolated hits if there is a hit and use that hit
	vector<int > imatches;
	size_t imatch=0;
	float glmatch=9999.;
	for (size_t iexp=0; iexp<expTrajMeasurements.size(); iexp++) {
	  const DetId & exphit_detId = expTrajMeasurements[iexp].recHit()->geographicalId();
	  int exphit_ladder = PXBDetId(exphit_detId).ladder();
	  int exphit_mod = PXBDetId(exphit_detId).module();
	  int dladder = abs( exphit_ladder - hit_ladder ); 
	  if (dladder > 10) dladder = 20 - dladder;
	  int dmodule = abs( exphit_mod - hit_mod);
	  if (dladder != 0 || dmodule != 0) {
	    continue;
	  }

	  TrajectoryStateOnSurface predTrajState=expTrajMeasurements[iexp].updatedState();
	  float x=predTrajState.globalPosition().x();
	  float y=predTrajState.globalPosition().y();
	  float z=predTrajState.globalPosition().z();
	  float dxyz=sqrt((chkx-x)*(chkx-x)+(chky-y)*(chky-y)+(chkz-z)*(chkz-z));

	  if (dxyz<=glmatch) {
	    glmatch=dxyz;
	    imatch=iexp;
	    imatches.push_back(int(imatch));
	  }

	} // found the propagated traj best matching the hit in data

	float lxmatch = 9999.0;
	float lymatch = 9999.0;
	if(!expTrajMeasurements.empty()){
	  if (glmatch<9999.) { // if there is any propagated trajectory for this hit
	    const DetId & matchhit_detId = expTrajMeasurements[imatch].recHit()->geographicalId();
      
	    int matchhit_ladder = PXBDetId(matchhit_detId).ladder();
	    int dladder = abs(matchhit_ladder-hit_ladder);
	    if (dladder > 10) dladder = 20 - dladder;
	    LocalPoint lp = expTrajMeasurements[imatch].updatedState().localPosition();
	    lxmatch=fabs(lp.x() - chklp.x());
	    lymatch=fabs(lp.y() - chklp.y());
	  }
	  if (lxmatch < maxlxmatch_ && lymatch < maxlymatch_) {
      
	    if (testhit->getType()!=TrackingRecHit::missing || keepOriginalMissingHit_) {
	      expTrajMeasurements.erase(expTrajMeasurements.begin()+imatch);
	    }
	    
	  }
	  
	} //expected trajectory measurment not empty
      }
    }//loop on trajectory measurments tmeasColl
    
    //if an extrapolated hit was found but not matched to an exisitng L1 hit then push the hit back into the collection
    //now keep the first one that is left
    if(!expTrajMeasurements.empty()){
      for (size_t f=0; f<expTrajMeasurements.size(); f++) {
	TrajectoryMeasurement AddHit=expTrajMeasurements[f];
	if (AddHit.recHit()->getType()==TrackingRecHit::missing){
	  tmeasColl.push_back(AddHit);
	  isBpixtrack = true;

	}
    
      }
    }

    if(isBpixtrack || isFpixtrack){
      if(trackref->pt()<0.6 ||
         nStripHits<8 ||
	 fabs(trackref->dxy(bestVtx->position()))>0.05 ||
	 fabs(trackref->dz(bestVtx->position()))>0.5) continue;
    
      if(debug_){
        std::cout << "isBpixtrack : " << isBpixtrack << std::endl;
        std::cout << "isFpixtrack : " << isFpixtrack << std::endl;
      }
      //std::cout<<"This tracks has so many hits: "<<tmeasColl.size()<<std::endl;
      for(std::vector<TrajectoryMeasurement>::const_iterator tmeasIt = tmeasColl.begin(); tmeasIt!=tmeasColl.end(); tmeasIt++){   
	//if(! tmeasIt->updatedState().isValid()) continue; 
	TrajectoryStateOnSurface tsos = tmeasIt->updatedState();

	TransientTrackingRecHit::ConstRecHitPointer hit = tmeasIt->recHit();
	if(hit->geographicalId().det() != DetId::Tracker )
	  continue; 
	else {
	  
	  // 	  //residual
      	  const DetId & hit_detId = hit->geographicalId();
	  //uint IntRawDetID = (hit_detId.rawId());
	  uint IntSubDetID = (hit_detId.subdetId());
	  
 	  if(IntSubDetID == 0 ){
	    if(debug_) std::cout << "NO IntSubDetID\n";
	    continue;
	  }
	  if(IntSubDetID!=PixelSubdetector::PixelBarrel && IntSubDetID!=PixelSubdetector::PixelEndcap)
	    continue;
	  
	  int disk=0; int layer=0; int panel=0; int module=0; bool isHalfModule=false;

	  const SiPixelRecHit* pixhit = dynamic_cast<const SiPixelRecHit*>(hit->hit());

	  if(IntSubDetID==PixelSubdetector::PixelBarrel){ // it's a BPIX hit
	    layer = PixelBarrelName(hit_detId,pTT,isUpgrade).layerName();
	    isHalfModule = PixelBarrelName(hit_detId,pTT,isUpgrade).isHalfModule();

	    if (hit->isValid()){ //fill the cluster probability in barrel
	      bool plus=true;
	      if ((PixelBarrelName(hit_detId,pTT,isUpgrade).shell()== PixelBarrelName::Shell::mO) || (PixelBarrelName(hit_detId,pTT,isUpgrade).shell()==PixelBarrelName::Shell::mI)) plus=false;	   
	      double clusterProbability= pixhit->clusterProbability(0);	     
	      if (clusterProbability!=0) fillClusterProbability(layer,0,plus,log10(clusterProbability));	      
	    }

	  }else if(IntSubDetID==PixelSubdetector::PixelEndcap){ // it's an FPIX hit
	    disk = PixelEndcapName(hit_detId,pTT,isUpgrade).diskName();
	    panel = PixelEndcapName(hit_detId,pTT,isUpgrade).pannelName();
	    module = PixelEndcapName(hit_detId,pTT,isUpgrade).plaquetteName();

	    if (hit->isValid()){
	      bool plus=true;
	      if ((PixelEndcapName(hit_detId,pTT,isUpgrade).halfCylinder()== PixelEndcapName::HalfCylinder::mO) || (PixelEndcapName(hit_detId,pTT,isUpgrade).halfCylinder()==PixelEndcapName::HalfCylinder::mI)) plus=false;
	      double clusterProbability= pixhit->clusterProbability(0);
	      if (clusterProbability!=0) fillClusterProbability(0,disk,plus,log10(clusterProbability));
	    }
          }
          	  
	  if(layer==1){
	    if(fabs(trackref->dxy(bestVtx->position()))>0.01 ||
	       fabs(trackref->dz(bestVtx->position()))>0.1) continue;
	    if(!(L2hits>0&&L3hits>0) && !(L2hits>0&&D1hits>0) && !(D1hits>0&&D2hits>0)) continue;
	  }else if(layer==2){
	    if(fabs(trackref->dxy(bestVtx->position()))>0.02 ||
	       fabs(trackref->dz(bestVtx->position()))>0.1) continue;
	    if(!(L1hits>0&&L3hits>0) && !(L1hits>0&&D1hits>0)) continue;
	  }else if(layer==3){
	    if(fabs(trackref->dxy(bestVtx->position()))>0.02 ||
	       fabs(trackref->dz(bestVtx->position()))>0.1) continue;
	    if(!(L1hits>0&&L2hits>0)) continue;
	  }else if(layer==4){
	    if(fabs(trackref->dxy(bestVtx->position()))>0.02 ||
	       fabs(trackref->dz(bestVtx->position()))>0.1) continue;
	  }else if(disk==1){
	    if(fabs(trackref->dxy(bestVtx->position()))>0.05 ||
	       fabs(trackref->dz(bestVtx->position()))>0.5) continue;
	    if(!(L1hits>0&&D2hits>0) && !(L2hits>0&&D2hits>0)) continue;
	  }else if(disk==2){
	    if(fabs(trackref->dxy(bestVtx->position()))>0.05 ||
	       fabs(trackref->dz(bestVtx->position()))>0.5) continue;
	    if(!(L1hits>0&&D1hits>0)) continue;
	  }else if(disk==3){
	    if(fabs(trackref->dxy(bestVtx->position()))>0.05 ||
	       fabs(trackref->dz(bestVtx->position()))>0.5) continue;
	  }
	  
	  //check whether hit is valid or missing using track algo flag
          bool isHitValid   =hit->hit()->getType()==TrackingRecHit::valid;
          bool isHitMissing =hit->hit()->getType()==TrackingRecHit::missing;
	  
	  if(debug_) std::cout << "the hit is persistent\n";
	 
	  std::map<uint32_t, SiPixelHitEfficiencyModule*>::iterator pxd = theSiPixelStructure.find((*hit).geographicalId().rawId());

	  // calculate alpha and beta from cluster position
	  LocalTrajectoryParameters ltp = tsos.localParameters();
	  //LocalVector localDir = ltp.momentum()/ltp.momentum().mag();
	      
	  //*************** Edge cut ********************
	  double lx=tsos.localPosition().x();
	  double ly=tsos.localPosition().y();

	  if(fabs(lx)>0.55 || fabs(ly)>3.0) continue;

	  bool passedFiducial=true;

	  // Module fiducials:
	  if(IntSubDetID==PixelSubdetector::PixelBarrel && fabs(ly)>=3.1) passedFiducial=false;
	  if(IntSubDetID==PixelSubdetector::PixelEndcap &&
	     !((panel==1 &&
		((module==1 && fabs(ly)<0.7) ||
		 ((module==2 && fabs(ly)<1.1) &&
		  !(disk==-1 && ly>0.8 && lx>0.2) &&
		  !(disk==1 && ly<-0.7 && lx>0.2) &&
		  !(disk==2 && ly<-0.8)) ||
		 ((module==3 && fabs(ly)<1.5) &&
		  !(disk==-2 && lx>0.1 && ly>1.0) &&
		  !(disk==2 && lx>0.1 && ly<-1.0)) ||
		 ((module==4 && fabs(ly)<1.9) &&
		  !(disk==-2 && ly>1.5) &&
		  !(disk==2 && ly<-1.5)))) ||
	       (panel==2 &&
		((module==1 && fabs(ly)<0.7) ||
		 (module==2 && fabs(ly)<1.2 &&
		  !(disk>0 && ly>1.1) &&
		  !(disk<0 && ly<-1.1)) ||
		 (module==3 && fabs(ly)<1.6 &&
		  !(disk>0 && ly>1.5) &&
		  !(disk<0 && ly<-1.5)))))) passedFiducial=false;
	  if(IntSubDetID==PixelSubdetector::PixelEndcap &&
	     ((panel==1 && (module==1 || (module>=3 && abs(disk)==1))) ||
	      (panel==2 && ((module==1 && abs(disk)==2) ||
			    (module==3 && abs(disk)==1))))) passedFiducial=false;
	  // ROC fiducials:
	  double ly_mod = fabs(ly);
	  if(IntSubDetID==PixelSubdetector::PixelEndcap && (panel+module)%2==1) ly_mod=ly_mod+0.405;
	  float d_rocedge = fabs(fmod(ly_mod,0.81)-0.405);
	  if(d_rocedge<=0.0625) passedFiducial=false;
	  if(!( (IntSubDetID==PixelSubdetector::PixelBarrel &&
		 ((!isHalfModule && fabs(lx)<0.6) ||
		  (isHalfModule && lx>-0.3 && lx<0.2))) ||
		(IntSubDetID==PixelSubdetector::PixelEndcap &&
		 ((panel==1 &&
		   ((module==1 && fabs(lx)<0.2) ||
		    (module==2 &&
		     ((fabs(lx)<0.55 && abs(disk)==1) ||
		      (lx>-0.5 && lx<0.2 && disk==-2) ||
		      (lx>-0.5 && lx<0.0 && disk==2))) ||
		    (module==3 && lx>-0.6 && lx<0.5) ||
		    (module==4 && lx>-0.3 && lx<0.15))) ||
		  (panel==2 &&
		   ((module==1 && fabs(lx)<0.6) ||
		    (module==2 &&
		     ((fabs(lx)<0.55 && abs(disk)==1) ||
		      (lx>-0.6 && lx<0.5 && abs(disk)==2))) ||
		    (module==3 && fabs(lx)<0.5))))))) passedFiducial=false;
	  if(((IntSubDetID==PixelSubdetector::PixelBarrel && !isHalfModule) || 
	      (IntSubDetID==PixelSubdetector::PixelEndcap && !(panel==1 && (module==1 || module==4)))) &&
	     fabs(lx)<0.06) passedFiducial=false;
	       
	       
	  //*************** find closest clusters ********************
	  float dx_cl[2]; float dy_cl[2]; dx_cl[0]=dx_cl[1]=dy_cl[0]=dy_cl[1]=-9999.;
	  ESHandle<PixelClusterParameterEstimator> cpEstimator;
	  iSetup.get<TkPixelCPERecord>().get("PixelCPEGeneric", cpEstimator);
	  if(cpEstimator.isValid()){
	    const PixelClusterParameterEstimator &cpe(*cpEstimator);
	    edm::ESHandle<TrackerGeometry> tracker;
	    iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
	    if(tracker.isValid()){
	      const TrackerGeometry *tkgeom=&(*tracker);
	      edm::Handle<edmNew::DetSetVector<SiPixelCluster> > clusterCollectionHandle;
	      iEvent.getByToken( clusterCollectionToken_, clusterCollectionHandle );
	      if(clusterCollectionHandle.isValid()){
		const edmNew::DetSetVector<SiPixelCluster>& clusterCollection=*clusterCollectionHandle;
		edmNew::DetSetVector<SiPixelCluster>::const_iterator itClusterSet=clusterCollection.begin();
		float minD[2]; minD[0]=minD[1]=10000.;
		for( ; itClusterSet!=clusterCollection.end(); itClusterSet++){
		  DetId detId(itClusterSet->id());
		  if(detId.rawId()!=hit->geographicalId().rawId()) continue;
		  //unsigned int sdId=detId.subdetId();
		  const PixelGeomDetUnit *pixdet=(const PixelGeomDetUnit*) tkgeom->idToDetUnit(detId);
		  edmNew::DetSet<SiPixelCluster>::const_iterator itCluster=itClusterSet->begin();
		  for( ; itCluster!=itClusterSet->end(); ++itCluster){
		    LocalPoint lp(itCluster->x(), itCluster->y(), 0.);
		    PixelClusterParameterEstimator::ReturnType params=cpe.getParameters(*itCluster,*pixdet);
		    lp=std::get<0>(params);
		    float D = sqrt((lp.x()-lx)*(lp.x()-lx)+(lp.y()-ly)*(lp.y()-ly));
		    if(D<minD[0]){
		      minD[1]=minD[0];
		      dx_cl[1]=dx_cl[0];
		      dy_cl[1]=dy_cl[0];
		      minD[0]=D;
		      dx_cl[0]=lp.x();
		      dy_cl[0]=lp.y();
		    }else if(D<minD[1]){
		      minD[1]=D;
		      dx_cl[1]=lp.x();
		      dy_cl[1]=lp.y();
		    }
		  }  // loop on clusterSets 
		} // loop on clusterCollection
		for(size_t i=0; i<2; i++){
		  if(minD[i]<9999.){
		    dx_cl[i]=fabs(dx_cl[i]-lx);
		    dy_cl[i]=fabs(dy_cl[i]-ly);
		  }
		}
	      } // valid clusterCollectionHandle
	    } // valid tracker
	  } // valid cpEstimator
	  // distance of hit from closest cluster!
	  float d_cl[2]; d_cl[0]=d_cl[1]=-9999.;
	  if(dx_cl[0]!=-9999. && dy_cl[0]!=-9999.) d_cl[0]=sqrt(dx_cl[0]*dx_cl[0]+dy_cl[0]*dy_cl[0]);
	  if(dx_cl[1]!=-9999. && dy_cl[1]!=-9999.) d_cl[1]=sqrt(dx_cl[1]*dx_cl[1]+dy_cl[1]*dy_cl[1]);
	  if(isHitMissing && (d_cl[0]<0.05 || d_cl[1]<0.05)){ isHitMissing=0; isHitValid=1; }	      
	      
	  if(debug_){
	    std::cout << "Ready to add hit in histogram:\n";
	    //std::cout << "detid: "<<hit_detId<<std::endl;
	    std::cout << "isHitValid: "<<isHitValid<<std::endl;
	    std::cout << "isHitMissing: "<<isHitMissing<<std::endl;
	    //std::cout << "passedEdgeCut: "<<passedFiducial<<std::endl;		
	  }    
	  
	  if (nStripHits<11) continue; //Efficiency plots are filled with hits on tracks that have at least 11 Strip hits 
    
	  if(pxd!=theSiPixelStructure.end() && isHitValid && passedFiducial)
	    ++nvalid;
	  if(pxd!=theSiPixelStructure.end() && isHitMissing && passedFiducial)
	    ++nmissing;
		
	  if (pxd!=theSiPixelStructure.end() && passedFiducial && (isHitValid || isHitMissing))
	    (*pxd).second->fill(pTT,ltp, isHitValid, modOn, ladOn, layOn, phiOn, bladeOn, diskOn, ringOn); 	

	  //}//end if (persistent hit exists and is pixel hit)
	  
	}//end of else 
	
	
      }//end for (all traj measurements of pixeltrack)
    }//end if (is pixeltrack)
    else
      if(debug_) std::cout << "no pixeltrack:\n";
    
  }//end loop on map entries
}


DEFINE_FWK_MODULE(SiPixelHitEfficiencySource); // define this as a plug-in 
