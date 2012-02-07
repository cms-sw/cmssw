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
#include <string>
#include <vector>
#include <utility>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"

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
  diskOn( pSet.getUntrackedParameter<bool>("diskOn",false) )
  //updateEfficiencies( pSet.getUntrackedParameter<bool>("updateEfficiencies",false) )
 { 
   pSet_ = pSet; 
   debug_ = pSet_.getUntrackedParameter<bool>("debug", false); 
   tracksrc_ = pSet_.getParameter<edm::InputTag>("trajectoryInput");
   applyEdgeCut_ = pSet_.getUntrackedParameter<bool>("applyEdgeCut");
   nSigma_EdgeCut_ = pSet_.getUntrackedParameter<double>("nSigma_EdgeCut");
   dbe_ = edm::Service<DQMStore>().operator->();

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

void SiPixelHitEfficiencySource::beginJob() {
  LogInfo("PixelDQM") << "SiPixelHitEfficiencySource beginJob()" << endl;
  firstRun = true;
}

void SiPixelHitEfficiencySource::beginRun(const edm::Run& r, edm::EventSetup const& iSetup) {
  LogInfo("PixelDQM") << "SiPixelHitEfficiencySource beginRun()" << endl;
  
  if(firstRun){
  // retrieve TrackerGeometry for pixel dets
  edm::ESHandle<TrackerGeometry> TG;
  iSetup.get<TrackerDigiGeometryRecord>().get(TG);
  if (debug_) LogVerbatim("PixelDQM") << "TrackerGeometry "<< &(*TG) <<" size is "<< TG->dets().size() << endl;
 
  // build theSiPixelStructure with the pixel barrel and endcap dets from TrackerGeometry
  for (TrackerGeometry::DetContainer::const_iterator pxb = TG->detsPXB().begin();  
       pxb!=TG->detsPXB().end(); pxb++) {
    if (dynamic_cast<PixelGeomDetUnit*>((*pxb))!=0) {
      SiPixelHitEfficiencyModule* module = new SiPixelHitEfficiencyModule((*pxb)->geographicalId().rawId());
      theSiPixelStructure.insert(pair<uint32_t, SiPixelHitEfficiencyModule*>((*pxb)->geographicalId().rawId(), module));
    }
  }
  for (TrackerGeometry::DetContainer::const_iterator pxf = TG->detsPXF().begin(); 
       pxf!=TG->detsPXF().end(); pxf++) {
    if (dynamic_cast<PixelGeomDetUnit*>((*pxf))!=0) {
      SiPixelHitEfficiencyModule* module = new SiPixelHitEfficiencyModule((*pxf)->geographicalId().rawId());
      theSiPixelStructure.insert(pair<uint32_t, SiPixelHitEfficiencyModule*>((*pxf)->geographicalId().rawId(), module));
    }
  }
  LogInfo("PixelDQM") << "SiPixelStructure size is " << theSiPixelStructure.size() << endl;

  // book residual histograms in theSiPixelFolder - one (x,y) pair of histograms per det
  SiPixelFolderOrganizer theSiPixelFolder;
  for (std::map<uint32_t, SiPixelHitEfficiencyModule*>::iterator pxd = theSiPixelStructure.begin(); 
       pxd!=theSiPixelStructure.end(); pxd++) {

    if(modOn){
      if (theSiPixelFolder.setModuleFolder((*pxd).first)) (*pxd).second->book(pSet_);
      else throw cms::Exception("LogicError") << "SiPixelHitEfficiencySource Folder Creation Failed! "; 
    }
    if(ladOn){
      if (theSiPixelFolder.setModuleFolder((*pxd).first,1)) (*pxd).second->book(pSet_,1);
      else throw cms::Exception("LogicError") << "SiPixelHitEfficiencySource ladder Folder Creation Failed! "; 
    }
    if(layOn){
      if (theSiPixelFolder.setModuleFolder((*pxd).first,2)) (*pxd).second->book(pSet_,2);
      else throw cms::Exception("LogicError") << "SiPixelHitEfficiencySource layer Folder Creation Failed! "; 
    }
    if(phiOn){
      if (theSiPixelFolder.setModuleFolder((*pxd).first,3)) (*pxd).second->book(pSet_,3);
      else throw cms::Exception("LogicError") << "SiPixelHitEfficiencySource phi Folder Creation Failed! "; 
    }
    if(bladeOn){
      if (theSiPixelFolder.setModuleFolder((*pxd).first,4)) (*pxd).second->book(pSet_,4);
      else throw cms::Exception("LogicError") << "SiPixelHitEfficiencySource Blade Folder Creation Failed! "; 
    }
    if(diskOn){
      if (theSiPixelFolder.setModuleFolder((*pxd).first,5)) (*pxd).second->book(pSet_,5);
      else throw cms::Exception("LogicError") << "SiPixelHitEfficiencySource Disk Folder Creation Failed! "; 
    }
    if(ringOn){
      if (theSiPixelFolder.setModuleFolder((*pxd).first,6)) (*pxd).second->book(pSet_,6);
      else throw cms::Exception("LogicError") << "SiPixelHitEfficiencySource Ring Folder Creation Failed! "; 
    }
  }
  
  nvalid=0;
  nmissing=0;
  
  firstRun = false;
  }
}


void SiPixelHitEfficiencySource::endJob(void) {
  LogInfo("PixelDQM") << "SiPixelHitEfficiencySource endJob()";

  // save the residual histograms to an output root file
  bool saveFile = pSet_.getUntrackedParameter<bool>("saveFile", true);
  if (saveFile) { 
    std::string outputFile = pSet_.getParameter<std::string>("outputFile");
    LogInfo("PixelDQM") << " - saving histograms to "<< outputFile.data();
    dbe_->save(outputFile);
  } 
  LogInfo("PixelDQM") << endl; // dbe_->showDirStructure();
  
  //std::cout<< "********** SUMMARY **********"<<std::endl;
  //std::cout<< "number of valid hits: "<<nvalid<<std::endl;
  //std::cout<< "number of missing hits: "<<nmissing<<std::endl;
}


void SiPixelHitEfficiencySource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<reco::VertexCollection> vertexCollectionHandle;
  iEvent.getByLabel("offlinePrimaryVertices", vertexCollectionHandle);
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
    
  //Get the geometry
  ESHandle<TrackerGeometry> TG;
  iSetup.get<TrackerDigiGeometryRecord>().get(TG);
  const TrackerGeometry* theTrackerGeometry = TG.product();
  
  //get the map
  edm::Handle<TrajTrackAssociationCollection> match;
  iEvent.getByLabel(tracksrc_,match);  
  const TrajTrackAssociationCollection ttac = *(match.product());

  if(debug_){
    //std::cout << "Trajectories\t : " << trajColl.size() << std::endl;
    //std::cout << "recoTracks  \t : " << trackColl.size() << std::endl;
    std::cout << "+++ NEW EVENT +++"<< std::endl;
    std::cout << "Map entries \t : " << ttac.size() << std::endl;
  }

  std::set<SiPixelCluster> clusterSet;
  TrajectoryStateCombiner tsoscomb;

  //Loop over map entries
  for(TrajTrackAssociationCollection::const_iterator it =  ttac.begin();it !=  ttac.end(); ++it){
    const edm::Ref<std::vector<Trajectory> > traj_iterator = it->key;  
    // Trajectory Map, extract Trajectory for this track
    reco::TrackRef trackref = it->val;
    //tracks++;
    bool isBpixtrack = false, isFpixtrack = false;
    int nStripHits=0; int L1hits=0; int L2hits=0; int L3hits=0; int D1hits=0; int D2hits=0;
    std::vector<TrajectoryMeasurement> tmeasColl =traj_iterator->measurements();
    std::vector<TrajectoryMeasurement>::const_iterator tmeasIt;
    //loop on measurements to find out what kind of hits there are
    for(tmeasIt = tmeasColl.begin();tmeasIt!=tmeasColl.end();tmeasIt++){
      //if(! tmeasIt->updatedState().isValid()) continue; NOT NECESSARY (I HOPE)
      TransientTrackingRecHit::ConstRecHitPointer testhit = tmeasIt->recHit();
      if(testhit->geographicalId().det() != DetId::Tracker) continue; 
      uint testSubDetID = (testhit->geographicalId().subdetId()); 
      const DetId & hit_detId = testhit->geographicalId();
      if(testSubDetID==PixelSubdetector::PixelBarrel){
        isBpixtrack = true;
        int layer = PixelBarrelName(hit_detId).layerName();
	if(layer==1) L1hits++;
	if(layer==2) L2hits++;
	if(layer==3) L3hits++;
      }
      if(testSubDetID==PixelSubdetector::PixelEndcap){
        isFpixtrack = true;
	int disk = PixelEndcapName(hit_detId).diskName();
	if(disk==1) D1hits++;
	if(disk==2) D2hits++;
      }
      if(testSubDetID==StripSubdetector::TIB) nStripHits++;
      if(testSubDetID==StripSubdetector::TOB) nStripHits++;
      if(testSubDetID==StripSubdetector::TID) nStripHits++;
      if(testSubDetID==StripSubdetector::TEC) nStripHits++;
    }
    if(isBpixtrack || isFpixtrack){
      if(trackref->pt()<0.6 ||
         nStripHits<11 ||
	 fabs(trackref->dxy(bestVtx->position()))>0.05 ||
	 fabs(trackref->dz(bestVtx->position()))>0.5) continue;
    
      if(debug_){
        std::cout << "isBpixtrack : " << isBpixtrack << std::endl;
        std::cout << "isFpixtrack : " << isFpixtrack << std::endl;
      }
      
      
      //std::cout<<"This track has so many hits: "<<tmeasColl.size()<<std::endl;
      // This is the main part, looping over the trajectory from the outside in:
      for(std::vector<TrajectoryMeasurement>::const_iterator tmeasIt = tmeasColl.begin(); tmeasIt!=tmeasColl.end(); tmeasIt++){   
	TrajectoryStateOnSurface tsos = tsoscomb( tmeasIt->forwardPredictedState(), tmeasIt->backwardPredictedState() );
	TransientTrackingRecHit::ConstRecHitPointer hit = tmeasIt->recHit();
	if(hit->geographicalId().det() != DetId::Tracker) continue; 	  
 	//residual
      	const DetId & hit_detId = hit->geographicalId();
	uint IntSubDetID = (hit_detId.subdetId());	  
 	if(IntSubDetID == 0 ){
	  if(debug_) std::cout << "NO IntSubDetID\n";
	  continue;
	}
	bool isBarrel = false; bool isEndcap = false;
	if(IntSubDetID==PixelSubdetector::PixelBarrel) isBarrel=true;
        if(IntSubDetID==PixelSubdetector::PixelEndcap) isEndcap=true;
	if(!isBarrel && !isEndcap) continue;
	int disk=0; int layer=0; int panel=0; int module=0; bool isHalfModule=false; int ladder=0;
	if(isBarrel){
          layer = PixelBarrelName(hit_detId).layerName();
	  isHalfModule = PixelBarrelName(hit_detId).isHalfModule();
	  module = PixelBarrelName(hit_detId).moduleName();
	  ladder = PixelBarrelName(hit_detId).ladderName();
	}else if(isEndcap){
	  disk = PixelEndcapName(hit_detId).diskName();
	  panel = PixelEndcapName(hit_detId).pannelName();
	  module = PixelEndcapName(hit_detId).plaquetteName();
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
	}else if(disk==1){
	  if(fabs(trackref->dxy(bestVtx->position()))>0.05 ||
	     fabs(trackref->dz(bestVtx->position()))>0.5) continue;
	  if(!(L1hits>0&&D2hits>0) && !(L2hits>0&&D2hits>0)) continue;
	}else if(disk==2){
	  if(fabs(trackref->dxy(bestVtx->position()))>0.05 ||
	     fabs(trackref->dz(bestVtx->position()))>0.5) continue;
	  if(!(L1hits>0&&D1hits>0)) continue;
	}
	//check whether hit is valid or missing using track algo flag
        bool isHitValid   =hit->hit()->getType()==TrackingRecHit::valid;
        bool isHitMissing =hit->hit()->getType()==TrackingRecHit::missing;
	//define tracker and pixel geometry and topology
	const TrackerGeometry& theTracker(*theTrackerGeometry);
	const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*> (theTracker.idToDet(hit_detId) );
	//test if PixelGeomDetUnit exists
	if(theGeomDet == 0) continue;
	std::map<uint32_t, SiPixelHitEfficiencyModule*>::iterator pxd = theSiPixelStructure.find((*hit).geographicalId().rawId());
        // calculate alpha and beta from cluster position
	LocalTrajectoryParameters ltp = tsos.localParameters();
	      
//////////////////////////////////////////////////////////	      
	// extrapolate L2 hits to L1 to reconstruct missing hits, which are not stored by default:
        std::vector<TrajectoryMeasurement> expTrajMeasurements; // for Layer1 only this gets recorded; will be filled because we first loop over L2 hits...
        int imeas=0;
	int realHitFound=0;
        double glx=tsos.globalPosition().x();
	double gly=tsos.globalPosition().y();
	double glz=tsos.globalPosition().z();
	LocalPoint chklp=chkPredTrajState.localPosition();
        if(isBarrel && layer==1){
	  // Pick a L1 hit; compare with extrapolated hit from L2; if they match in position, remove L1 hit from extrapolated hit
	  // list, because it's a valid hit! Whatever hits are remaining in the end in the extrapolated list, are the missing hits 
	  // in L1!
	  size_t imatch=0;
	  double glmatch=9999.;
	  int explayer=0; bool expisHalfModule=false; int expladder=0; int expmodule=0;
	  int dladder=9999; int dmodule=9999;
	  for(size_t iexp=0; iexp<expTrajMeasurements.size(); iexp++){
	    const DetId & exphit_detId = expTrajMeasurements[iexp].recHit()->geographicalId();
	    uint expIntSubDetID = (exphit_detId.subdetId());	  
 	    if(expIntSubDetID == 0 ) continue;
	    if(expIntSubDetID!=PixelSubdetector::PixelBarrel && expIntSubDetID!=PixelSubdetector::PixelEndcap) continue;	  
	    if(expIntSubDetID==PixelSubdetector::PixelBarrel){ // it's a BPIX hit
              explayer = PixelBarrelName(exphit_detId).layerName();
	      expladder = PixelBarrelName(exphit_detId).ladderName();
	      expmodule = PixelBarrelName(exphit_detId).moduleName();
	    }
	    if(explayer!=1) continue;
	    dladder=abs(expladder-ladder);
	    if(dladder>10) dladder=20-dladder;
	    dmodule=abs(expmodule-module);
	    if(dladder!=0 || dmodule!=0) continue; // module mismatch
	    TrajectoryStateOnSurface predTrajState=expTrajMeasurements[iexp].updatedState();
	    double x=predTrajState.globalPosition().x();
	    double y=predTrajState.globalPosition().y();
	    double z=predTrajState.globalPosition().z();
	    double dxyz=sqrt((glx-x)*(glx-x)+(gly-y)*(gly-y)+(glz-z)*(glz-z));
	    if(dxyz<glmatch){
	      glmatch=dxyz;
	      imatch=iexp;
	    }
	  } // found the propagated traj best matching the hit in data
	  double lxmatch=9999.; double lymatch=9999.; double lxymatch=9999.;
	  if(glmatch<9999.){ // if there is any propagated trajectory for this hit
	    LocalPoint lp=expTrajMeasurements[imatch].updatedState().localPosition();
	    lxmatch=fabs(lp.x()-chklp.x());
	    lymatch=fabs(lp.y()-chklp.y());
	    lxymatch=sqrt(lxmatch*lxmatch+lymatch*lymatch);
	  }	  
	  if(lxmatch<0.2 && lymatch<0.2) expTrajMeasurements.erase(expTrajMeasurements.begin()+imatch);
	} // END of processing extrapolated hit
        // Propagate valid layer2 or disk1 hit:
	bool lastValidL2=false;
	if((isBarrel && layer==2) || (isEndcap && disk==1)){
	  if(isHitValid){
	    if(tmeasIt==tmeasColl.end()-1){
	      lastValidL2=true;
	    }else{
	      tmeasIt++;
	      TransientTrackingRecHit::ConstRecHitPointer nextRecHit = tmeasIt->recHit();
	      const DetId & nexthit_detId = nextRecHit->geographicalId();
	      int nextlayer=0;
	      uint nextIntSubDetID = (nexthit_detId.subdetId());	  
 	      if(nextIntSubDetID == 0 ) continue;
	      if(nextIntSubDetID!=PixelSubdetector::PixelBarrel && nextIntSubDetID!=PixelSubdetector::PixelEndcap) continue;	  
	      if(nextIntSubDetID==PixelSubdetector::PixelBarrel){ // it's a BPIX hit
                nextlayer = PixelBarrelName(nexthit_detId).layerName();
	      }
	      if(nextlayer==1) lastValidL2=true; 
	      tmeasIt--;
	    }
	  }
	}
	// found the matching last valid hit in L2 that was extrapolated into L1:
	if(lastValidL2){
	  std::vector< BarrelDetLayer*> pxbLayers = measurementTrackerHandle->geometricSearchTracker()->pixelBarrelLayers();
	  const DetLayer* pxb1 = pxbLayers[0];
	  const MeasurementEstimator* estimator = est.product();
	  const LayerMeasurements* theLayerMeasurements = new LayerMeasurements(&*measurementTrackerHandle);
	  const TrajectoryStateOnSurface tsosPXB2 = tmeasIt->updatedState();
	  expTrajMeasurements = theLayerMeasurements->measurements(*pxb1, tsosPXB2, *thePropagator, *estimator);
	  delete theLayerMeasurements;
	  if(!expTrajMeasurements.empty()){
	    TrajectoryMeasurement pxb1TM(expTrajMeasurements.front());
	    ConstReferenceCountingPointer<TransientTrackingRecHit> pxb1Hit;
	    pxb1Hit = pxb1TM.recHit();	    
	    if(pxb1Hit->geographicalId().rawId()!=0){
	          ModuleData pxbMod=getModuleData(pxb1Hit->geographicalId().rawId(), federrors, "offline");
	    }
	    if(isBarrel){
              layer = PixelBarrelName(pxb1Hit->geographicalId()).layerName();
	      module = PixelBarrelName(pxb1Hit->geographicalId()).moduleName();
	      ladder = PixelBarrelName(pxb1Hit->geographicalId()).ladderName();
	    }else if(isEndcap){
	      disk = PixelEndcapName(pxb1Hit->geographicalId()).diskName();
	      panel = PixelEndcapName(pxb1Hit->geographicalId()).pannelName();
	      module = PixelEndcapName(pxb1Hit->geographicalId()).plaquetteName();
	    }
	  }
	}
	imeas++;
	correctHitTypeAssignment(meas, recHit); // Needs to have meas.mod_on set correctly
        if(recHit->getType()==TrackingRecHit::valid){ isHitValid=true; isHitMissing=false; }
        if(recHit->getType()==TrackingRecHit::missing){ isHitMissing=true; isHitValid=false; }
        // Exceptions:
        // One full module on layer 1 is out. Need to fix classification here, because
        // the trajectory propagation does not include this
        if(isBarrel && layer==1 && ladder==9 && module==2){
          isHitValid=false;
	  isHitMissing=false;
        }
        // One full module on layer 2, because it is not in the database yet
        if(isBarrel && layer==2 && ladder==-8 && module==-4){
          isHitValid=false;
	  isHitMissing=false;
        }

////////////////////////////////////////////	      
	      
	      
        //*************** Edge cut ********************
	double lx=tsos.localPosition().x();
	double ly=tsos.localPosition().y();
	if(fabs(lx)>0.55 || fabs(ly)>3.0) continue;
	bool passedFiducial=true;
	// Module fiducials:
	if(isBarrel && fabs(ly)>=3.1) passedFiducial=false;
	if(isEndcap &&
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
        if(isEndcap &&
	   ((panel==1 && (module==1 || (module>=3 && abs(disk)==1))) ||
	    (panel==2 && ((module==1 && abs(disk)==2) ||
		          (module==3 && abs(disk)==1))))) passedFiducial=false;
        // ROC fiducials:
	double ly_mod = fabs(ly);
	if(isEndcap && (panel+module)%2==1) ly_mod=ly_mod+0.405;
	float d_rocedge = fabs(fmod(ly_mod,0.81)-0.405);
	if(d_rocedge<=0.0625) passedFiducial=false;
	if(!((isBarrel &&
	      ((!isHalfModule && fabs(lx)<0.6) ||
		(isHalfModule && lx>-0.3 && lx<0.2))) ||
	     (isEndcap &&
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
        if(((isBarrel && !isHalfModule) || 
	    (isEndcap && !(panel==1 && (module==1 || module==4)))) &&
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
	    iEvent.getByLabel("siPixelClusters", clusterCollectionHandle);
	    if(clusterCollectionHandle.isValid()){
	      const edmNew::DetSetVector<SiPixelCluster>& clusterCollection=*clusterCollectionHandle;
	      edmNew::DetSetVector<SiPixelCluster>::const_iterator itClusterSet=clusterCollection.begin();
	      float minD[2]; minD[0]=minD[1]=10000.;
	      for( ; itClusterSet!=clusterCollection.end(); itClusterSet++){
		DetId detId(itClusterSet->id());
		if(detId.rawId()!=hit->geographicalId().rawId()) continue;
		const PixelGeomDetUnit *pixdet=(const PixelGeomDetUnit*) tkgeom->idToDetUnit(detId);
		edmNew::DetSet<SiPixelCluster>::const_iterator itCluster=itClusterSet->begin();
	        for( ; itCluster!=itClusterSet->end(); ++itCluster){
		  LocalPoint lp(itCluster->x(), itCluster->y(), 0.);
	          PixelClusterParameterEstimator::LocalValues params=cpe.localParameters(*itCluster,*pixdet);
	          lp=params.first;
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
	      	      
	if(pxd!=theSiPixelStructure.end() && isHitValid && passedFiducial) ++nvalid;
	if(pxd!=theSiPixelStructure.end() && isHitMissing && passedFiducial) ++nmissing;
		
	if(pxd!=theSiPixelStructure.end() && passedFiducial && (isHitValid || isHitMissing))
	  (*pxd).second->fill(ltp, isHitValid, modOn, ladOn, layOn, phiOn, bladeOn, diskOn, ringOn); 	
	  
      }//end of else 
    }//end for (all traj measurements of pixeltrack)
  }//end if (is pixeltrack)
}//end loop on map entries
}


DEFINE_FWK_MODULE(SiPixelHitEfficiencySource); // define this as a plug-in 
