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

#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

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
    std::vector<TrajectoryMeasurement> tmeasColl =traj_iterator->measurements();
    std::vector<TrajectoryMeasurement>::const_iterator tmeasIt;
    //loop on measurements to find out whether there are bpix and/or fpix hits
    for(tmeasIt = tmeasColl.begin();tmeasIt!=tmeasColl.end();tmeasIt++){
      //if(! tmeasIt->updatedState().isValid()) continue; NOT NECESSARY (I HOPE)
      TransientTrackingRecHit::ConstRecHitPointer testhit = tmeasIt->recHit();
      if(testhit->geographicalId().det() != DetId::Tracker) continue; 
      uint testSubDetID = (testhit->geographicalId().subdetId()); 
      if(testSubDetID==PixelSubdetector::PixelBarrel) isBpixtrack = true;
      if(testSubDetID==PixelSubdetector::PixelEndcap) isFpixtrack = true;
    }
    if(isBpixtrack || isFpixtrack){
    
      if(debug_){
        std::cout << "isBpixtrack : " << isBpixtrack << std::endl;
        std::cout << "isFpixtrack : " << isFpixtrack << std::endl;
      }
      for(std::vector<TrajectoryMeasurement>::const_iterator tmeasIt = tmeasColl.begin(); tmeasIt!=tmeasColl.end(); tmeasIt++){   
	//if(! tmeasIt->updatedState().isValid()) continue; 
	
	TrajectoryStateOnSurface tsos = tsoscomb( tmeasIt->forwardPredictedState(), tmeasIt->backwardPredictedState() );
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
	  
	  
	      //check wether hit is valid or missing using track algo flag
          bool isHitValid   =hit->hit()->getType()==TrackingRecHit::valid;
          bool isHitMissing =hit->hit()->getType()==TrackingRecHit::missing;
          //std::cout<<"------ New Hit"<<std::endl;
          //std::cout<<(hit->hit()->getType()==TrackingRecHit::missing)<<std::endl;
	  
	  // get the enclosed persistent hit
	  //const TrackingRecHit *persistentHit = hit->hit();
	  // check if it's not null, and if it's a valid pixel hit
	  //if ((persistentHit != 0) && (typeid(*persistentHit) == typeid(SiPixelRecHit))) {
	  
	    if(debug_) std::cout << "the hit is persistent\n";
	    
	    // tell the C++ compiler that the hit is a pixel hit
	    //const SiPixelRecHit* pixhit = dynamic_cast<const SiPixelRecHit*>( hit->hit() );
	 
	    //define tracker and pixel geometry and topology
	    const TrackerGeometry& theTracker(*theTrackerGeometry);
	    const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*> (theTracker.idToDet(hit_detId) );
	    //test if PixelGeomDetUnit exists
	    if(theGeomDet == 0) {
	      if(debug_) std::cout << "NO THEGEOMDET\n";
	      continue;
	    }
	      	      
	      //const RectangularPixelTopology * topol = dynamic_cast<const RectangularPixelTopology*>(&(theGeomDet->specificTopology()));

	      std::map<uint32_t, SiPixelHitEfficiencyModule*>::iterator pxd = theSiPixelStructure.find((*hit).geographicalId().rawId());

	      // calculate alpha and beta from cluster position
	      LocalTrajectoryParameters ltp = tsos.localParameters();
	      //LocalVector localDir = ltp.momentum()/ltp.momentum().mag();
	      
	      //float clust_alpha = atan2(localDir.z(), localDir.x());
	      //float clust_beta = atan2(localDir.z(), localDir.y());
	      
	      
	      
	      // THE CUTS
	      int nrows = theGeomDet->specificTopology().nrows();
	      int ncols = theGeomDet->specificTopology().ncolumns();
	      //
	      //std::pair<float,float> pitchTest = theGeomDet->specificTopology().pitch();
	      //RectangularPixelTopology rectTopolTest = RectangularPixelTopology(nrows, ncols, pitch.first, pitch.second);
	      //std::pair<float,float> pixelTest = rectTopol.pixel(tsos.localPosition());
	      //
	      std::pair<float,float> pixel = theGeomDet->specificTopology().pixel(tsos.localPosition());

	      //*************** Edge cut ********************
	      if(applyEdgeCut_){
	        bool passedEdgeCut = false; 
	        double nsigma = nSigma_EdgeCut_;
	      
	        //transforming localX,Y coordinates into a number of rows/columns
	        LocalPoint  actual = LocalPoint( tsos.localPosition().x(),tsos.localPosition().y() );
	        std::pair<float,float> pixelActual = theGeomDet->specificTopology().pixel(actual);
	      
	        //Adding/substracting nsigma times the error position to the acual traj prediction position
	        //Depending on the signes of localX,Y => 4 "quadrants"
	        LocalPoint  exagerated;
	        LocalError tsosErr = tsos.localError().positionError();	 
	        if ( tsos.localPosition().x()>0 && tsos.localPosition().y()>0 )
	          exagerated = LocalPoint(tsos.localPosition().x()+nsigma*std::sqrt(tsosErr.xx()),tsos.localPosition().y()+nsigma*std::sqrt(tsosErr.yy()) );
	        if ( tsos.localPosition().x()<0 && tsos.localPosition().y()>0)
	          exagerated = LocalPoint(tsos.localPosition().x()-nsigma*std::sqrt(tsosErr.xx()),tsos.localPosition().y()+nsigma*std::sqrt(tsosErr.yy())  );
	        if ( tsos.localPosition().x()<0 && tsos.localPosition().y()<0)
	          exagerated = LocalPoint(tsos.localPosition().x()-nsigma*std::sqrt(tsosErr.xx()),tsos.localPosition().y()-nsigma*std::sqrt(tsosErr.yy()) );
	        if ( tsos.localPosition().x()>0 && tsos.localPosition().y()<0)
	          exagerated = LocalPoint(tsos.localPosition().x()+nsigma*std::sqrt(tsosErr.xx()),tsos.localPosition().y()-nsigma*std::sqrt(tsosErr.yy()) );
	      
	        //transforming localX,Y coordinates into a number of rows/columns
	        std::pair<float,float> pixelExagerated = theGeomDet->specificTopology().pixel(exagerated);
	 
	        //if the modified traj prediction falls out of the module, we cut
	        if( pixelExagerated.first>nrows  || pixelExagerated.first<0)
	          passedEdgeCut = false;
	        if( pixelExagerated.second>ncols || pixelExagerated.second<0)
	          passedEdgeCut = false;
	      
	        if(!passedEdgeCut)
	          continue;
	      }
	      //*************** Telescope cut ********************
	      //not needed for collisions !!
	      
	      
	      
	      
	      
	      
	      
	      
	      if(debug_){
	        std::cout << "Ready to add hit in histogram:\n";
	        //std::cout << "detid: "<<hit_detId<<std::endl;
	        std::cout << "isHitValid: "<<isHitValid<<std::endl;
	        std::cout << "isHitMissing: "<<isHitMissing<<std::endl;
	        //std::cout << "passedEdgeCut: "<<passedEdgeCut<<std::endl;		
	      }
	      
	      
	      if(pxd!=theSiPixelStructure.end() && isHitValid)
	        ++nvalid;
	      if(pxd!=theSiPixelStructure.end() && isHitMissing)
	        ++nmissing;
		
	      if (pxd!=theSiPixelStructure.end() && (isHitValid || isHitMissing))
	        (*pxd).second->fill(ltp, isHitValid, modOn, ladOn, layOn, phiOn, bladeOn, diskOn, ringOn); 	

	  //}//end if (persistent hit exists and is pixel hit)
	  
	}//end of else 
	
	
      }//end for (all traj measurements of pixeltrack)
    }//end if (is pixeltrack)
    else
      if(debug_) std::cout << "no pixeltrack:\n";
    
  }//end loop on map entries
}


DEFINE_FWK_MODULE(SiPixelHitEfficiencySource); // define this as a plug-in 
