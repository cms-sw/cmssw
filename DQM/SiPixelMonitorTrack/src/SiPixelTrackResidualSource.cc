// Package:    SiPixelMonitorTrack
// Class:      SiPixelTrackResidualSource
// 
// class SiPixelTrackResidualSource SiPixelTrackResidualSource.cc 
//       DQM/SiPixelMonitorTrack/src/SiPixelTrackResidualSource.cc
//
// Description: SiPixel hit-to-track residual data quality monitoring modules
// Implementation: prototype -> improved -> never final - end of the 1st step 
//
// Original Author: Shan-Huei Chuang
//         Created: Fri Mar 23 18:41:42 CET 2007
// $Id: SiPixelTrackResidualSource.cc,v 1.8 2008/03/01 20:19:52 lat Exp $


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

#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
#include "DQM/SiPixelMonitorTrack/interface/SiPixelTrackResidualSource.h"


using namespace std;
using namespace edm;


SiPixelTrackResidualSource::SiPixelTrackResidualSource(const edm::ParameterSet& pSet) { 
   pSet_ = pSet; 
  debug_ = pSet_.getUntrackedParameter<bool>("debug", false); 
    src_ = pSet_.getParameter<edm::InputTag>("src"); 
    dbe_ = edm::Service<DQMStore>().operator->();

  LogInfo("PixelDQM") << "SiPixelTrackResidualSource constructor" << endl;
}


SiPixelTrackResidualSource::~SiPixelTrackResidualSource() {
  LogInfo("PixelDQM") << "SiPixelTrackResidualSource destructor" << endl;
}


void SiPixelTrackResidualSource::beginJob(edm::EventSetup const& iSetup) {
  LogInfo("PixelDQM") << "SiPixelTrackResidualSource beginJob()" << endl;

  // retrieve the TrackerGeometry for pixel dets...
  edm::ESHandle<TrackerGeometry> TG;
  iSetup.get<TrackerDigiGeometryRecord>().get(TG);
  if (debug_) LogVerbatim("PixelDQM") << "TrackerGeometry "<< &(*TG) <<" size is "<< TG->dets().size() << endl;
 
  // build theSiPixelStructure with the pixel barrel and endcap dets in the TrackerGeometry...
  for (TrackerGeometry::DetContainer::const_iterator pxb = TG->detsPXB().begin();  
       pxb!=TG->detsPXB().end(); pxb++) {
    if (dynamic_cast<PixelGeomDetUnit*>((*pxb))!=0) {
      SiPixelTrackResidualModule* module = new SiPixelTrackResidualModule((*pxb)->geographicalId().rawId());
      theSiPixelStructure.insert(pair<uint32_t, SiPixelTrackResidualModule*>((*pxb)->geographicalId().rawId(), module));
    }
  }
  for (TrackerGeometry::DetContainer::const_iterator pxf = TG->detsPXF().begin(); 
       pxf!=TG->detsPXF().end(); pxf++) {
    if (dynamic_cast<PixelGeomDetUnit*>((*pxf))!=0) {
      SiPixelTrackResidualModule* module = new SiPixelTrackResidualModule((*pxf)->geographicalId().rawId());
      theSiPixelStructure.insert(pair<uint32_t, SiPixelTrackResidualModule*>((*pxf)->geographicalId().rawId(), module));
    }
  }
  LogInfo("PixelDQM") << "SiPixelStructure size is " << theSiPixelStructure.size() << endl;

  dbe_->setVerbose(0);

  // book residual histograms in theSiPixelFolder - 1 (x,y)-pair of histograms per det...
  SiPixelFolderOrganizer theSiPixelFolder;
  for (std::map<uint32_t, SiPixelTrackResidualModule*>::iterator pxd = theSiPixelStructure.begin(); 
       pxd!=theSiPixelStructure.end(); pxd++) {
    if (theSiPixelFolder.setModuleFolder((*pxd).first)) (*pxd).second->book(pSet_);
    else throw cms::Exception("LogicError") << "SiPixelTrackResidualSource Folder Creation Failed! "; 
  }
  if (debug_) {
    // book summary residual histograms in a debugging folder - 1 (x,y)-pair of histograms per subdetector... 
    dbe_->setCurrentFolder("debugging"); 
    char hisID[80]; 
    for (int s=0; s<3; s++) {
      sprintf(hisID,"residual_x_subdet_%i",s); 
      meSubdetResidualX[s] = dbe_->book1D(hisID,"Pixel Hit-to-Track Residual in X",500,-5.,5.);
    
      sprintf(hisID,"residual_y_subdet_%i",s); 
      meSubdetResidualY[s] = dbe_->book1D(hisID,"Pixel Hit-to-Track Residual in Y",500,-5.,5.);  
    }
  } 
}


void SiPixelTrackResidualSource::endJob(void) {
  LogInfo("PixelDQM") << "SiPixelTrackResidualSource endJob()";

  // save the residual histograms to an output root file...
  bool saveFile = pSet_.getUntrackedParameter<bool>("saveFile", true);
  if (saveFile) { 
    std::string outputFilename = pSet_.getParameter<std::string>("outputFilename");
    LogInfo("PixelDQM") << " - saving histograms to "<< outputFilename.data();
    dbe_->save(outputFilename);
  } 
  LogInfo("PixelDQM") << endl; // dbe_->showDirStructure();
}


void SiPixelTrackResidualSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // retrieve the TrackerGeometry again and the (Ideal)MagneticField 
  // for use in transforming a TrackCandidate's P(ersistent)TrajectoryStateoOnDet (PTSoD) to a TrajectoryStateOnSurface (TSoS)
  ESHandle<TrackerGeometry> TG;
  iSetup.get<TrackerDigiGeometryRecord>().get(TG);
  const TrackerGeometry* theTrackerGeometry = TG.product();
  
  ESHandle<MagneticField> MF;
  iSetup.get<IdealMagneticFieldRecord>().get(MF);
  const MagneticField* theMagneticField = MF.product();
  
  // retrieve the TransientTrackingRecHitBuilder to build TTRHs with TrackCandidate's TrackingRecHits for theFitter 
  std::string TTRHBuilder = pSet_.getParameter<std::string>("TTRHBuilder"); 
  ESHandle<TransientTrackingRecHitBuilder> TTRHB; 
  iSetup.get<TransientRecHitRecord>().get(TTRHBuilder, TTRHB);
  const TransientTrackingRecHitBuilder* theTTRHBuilder = TTRHB.product();
   
  // get a fitter to refit TrackCandidates, the same fitter as used in standard reconstruction 
  std::string Fitter = pSet_.getParameter<std::string>("Fitter");
  ESHandle<TrajectoryFitter> TF;
  iSetup.get<TrackingComponentsRecord>().get(Fitter, TF);
  const TrajectoryFitter* theFitter = TF.product();

  // get the TrackCandidateCollection... matching theFitter in the cfi, i.e. rs-RS, ckf-KF... 
  std::string TrackCandidateLabel = pSet_.getParameter<std::string>("TrackCandidateLabel");
  std::string TrackCandidateProducer = pSet_.getParameter<std::string>("TrackCandidateProducer");  
  Handle<TrackCandidateCollection> trackCandidateCollection;
  iEvent.getByLabel(TrackCandidateProducer, TrackCandidateLabel, trackCandidateCollection);

  for (TrackCandidateCollection::const_iterator tc = trackCandidateCollection->begin(); 
       tc!=trackCandidateCollection->end(); ++tc) {
    TrajectoryStateTransform transformer; 
    PTrajectoryStateOnDet tcPTSoD = tc->trajectoryStateOnDet();
    TrajectoryStateOnSurface tcTSoS = transformer.transientState(tcPTSoD, &(theTrackerGeometry->idToDet(tcPTSoD.detId())->surface()), 
						                 theMagneticField);
    const TrajectorySeed& tcSeed = tc->seed();

    const TrackCandidate::range& tcRecHits = tc->recHits();    
    if (debug_) cout << "track candidate has "<< int(tcRecHits.second - tcRecHits.first) <<" hits with ID "; 
    
    Trajectory::RecHitContainer tcTTRHs;
    for (TrackingRecHitCollection::const_iterator tcRecHit = tcRecHits.first; 
         tcRecHit!=tcRecHits.second; ++tcRecHit) { 
      if (debug_) cout << tcRecHit->geographicalId().rawId() <<" "; 
      
      tcTTRHs.push_back(theTTRHBuilder->build(&(*tcRecHit)));
    } 
    // note a TrackCandidate keeps only the PTSoD of the first hit as well as the seed and the hits. 
    // To 99.99%-recover all the hit's TSoSes, do a refit with the seed, the hits and an initial TSoS from the PTSoD 
    // and get a Trajectory that consists of all the hit's TrajectoryMeasurements (TMs). 
    std::vector<Trajectory> refitTrajectoryCollection = theFitter->fit(tcSeed, tcTTRHs, tcTSoS);	    
    if (debug_) cout << "refitTrajectoryCollection size is "<< refitTrajectoryCollection.size() << endl;

    if (refitTrajectoryCollection.size()>0) { // either 0 or 1... 
      const Trajectory& refitTrajectory = refitTrajectoryCollection.front();

      // retrieve and loop over all the TMs... 
      Trajectory::DataContainer refitTMs = refitTrajectory.measurements();								
      if (debug_) cout << "refitTrajectory has "<< refitTMs.size() <<" hits with ID "; 

      for (Trajectory::DataContainer::iterator refitTM = refitTMs.begin(); 
           refitTM!=refitTMs.end(); refitTM++) {  					
    	TransientTrackingRecHit::ConstRecHitPointer refitTTRH = refitTM->recHit();
        if (debug_) cout << refitTTRH->geographicalId().rawId() <<" "; 
	
	// only analyze the most individual pixel hit's TMs to calculate residuals... 
	const GeomDet* ttrhDet = refitTTRH->det(); 
	if (ttrhDet->components().empty() && (ttrhDet->subDetector()==GeomDetEnumerators::PixelBarrel ||				
    					      ttrhDet->subDetector()==GeomDetEnumerators::PixelEndcap)) {				

    	  // combine the forward and backward states without using the hit's information (hence unbiased by the hit); 
	  // the TM's updated state keeps the state combined and updated with the hit's info but we don't use the updated state at all 
	  TrajectoryStateOnSurface combinedTSoS = TrajectoryStateCombiner().combine(refitTM->forwardPredictedState(),        
    					        				    refitTM->backwardPredictedState());      
	  if (refitTTRH->isValid() && combinedTSoS.isValid()) { 
	    // calculate the distance between the hit location and the track-crossing point predicted by the combined state... 
            const GeomDetUnit* GDU = dynamic_cast<const GeomDetUnit*>(ttrhDet);
	    const Topology* theTopology = &(GDU->topology()); 									
    	    
	    MeasurementPoint hitPosition = theTopology->measurementPosition(refitTTRH->localPosition());				
    	    MeasurementPoint combinedTSoSPosition = theTopology->measurementPosition(combinedTSoS.localPosition());	
    	    
	    Measurement2DVector residual = hitPosition - combinedTSoSPosition;  						
    																
    	    // fill the residual histograms... 
	    std::map<uint32_t, SiPixelTrackResidualModule*>::iterator pxd = theSiPixelStructure.find(refitTTRH->geographicalId().rawId());	
    	    if (pxd!=theSiPixelStructure.end()) (*pxd).second->fill(residual); 				
    																
    	    if (debug_) {
	      if (ttrhDet->subDetector()==GeomDetEnumerators::PixelBarrel) {			        		       
    	        meSubdetResidualX[0]->Fill(residual.x());					        		   
    	        meSubdetResidualY[0]->Fill(residual.y());					        		   
    	      } 										        		       
    	      else {										  
    	        meSubdetResidualX[PXFDetId(refitTTRH->geographicalId()).side()]->Fill(residual.x());  						    
    	        meSubdetResidualY[PXFDetId(refitTTRH->geographicalId()).side()]->Fill(residual.y());  						    
    	      } 
	    }															
    	  }
    	} 											
      } 
      if (debug_) cout << endl; 															
    }																
  } 
}


DEFINE_FWK_MODULE(SiPixelTrackResidualSource); // define this as a plug-in 
