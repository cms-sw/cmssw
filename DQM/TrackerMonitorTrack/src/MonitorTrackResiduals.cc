#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/TrackerMonitorTrack/interface/MonitorTrackResiduals.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "RecoTracker/TrackProducer/interface/TrackingRecHitLessFromGlobalPosition.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"

MonitorTrackResiduals::MonitorTrackResiduals(const edm::ParameterSet& iConfig) {
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  conf_ = iConfig;
}

MonitorTrackResiduals::~MonitorTrackResiduals() { }

void MonitorTrackResiduals::beginJob(edm::EventSetup const& iSetup) {
  using namespace edm;

  // use SistripHistoId for producing histogram id (and title)
  SiStripHistoId hidmanager;
  // create SiStripFolderOrganizer
  SiStripFolderOrganizer folder_organizer;
  folder_organizer.setSiStripFolder(); // top SiStrip folder

  // take from eventSetup the SiStripDetCabling object

  edm::ESHandle<SiStripDetCabling> tkmechstruct;
  iSetup.get<SiStripDetCablingRcd>().get(tkmechstruct);

  // get list of active detectors from SiStripDetCabling
  std::vector<uint32_t> activeDets;
  activeDets.clear(); // just in case
  tkmechstruct->addActiveDetectorsRawIds(activeDets);

  // use SiStripSubStructure for selecting certain regions
  SiStripSubStructure substructure;
  std::vector<uint32_t> DetIds = activeDets;
  
    
    // book histo per each detector module
  for (std::vector<uint32_t>::const_iterator DetItr=activeDets.begin(); DetItr!=activeDets.end(); DetItr++)
    {
      folder_organizer.setDetectorFolder(*DetItr); // pas detid - uint32 to this method - sets appropriate detector folder
      int ModuleID = (*DetItr);
      folder_organizer.setDetectorFolder(*DetItr); // top Mechanical View Folder
      std::string hid = hidmanager.createHistoId("HitResiduals","det",*DetItr);
      HitResidual[ModuleID] = dbe->book1D(hid, hid, 50, -5., 5.);
      HitResidual[ModuleID]->setAxisTitle("Hit residuals on tracks crossing this detector module");
    }
	

}

void MonitorTrackResiduals::endJob(void) {
  dbe->showDirStructure();
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe->save(outputFileName);
  }
}


void MonitorTrackResiduals::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::string TrackCandidateProducer = conf_.getParameter<std::string>("TrackCandidateProducer");
  std::string TrackCandidateLabel = conf_.getParameter<std::string>("TrackCandidateLabel");

  Handle<std::vector<Trajectory> > trajCollectionHandle;
  iEvent.getByLabel(TrackCandidateProducer, TrackCandidateLabel,trajCollectionHandle);

  for(std::vector<Trajectory>::const_iterator it = trajCollectionHandle->begin(); it!=trajCollectionHandle->end();it++)
    {
      std::vector<TrajectoryMeasurement> tmColl = it->measurements();
      for(std::vector<TrajectoryMeasurement>::const_iterator itTraj = tmColl.begin(); itTraj!=tmColl.end(); itTraj++)
	{
	  if(! itTraj->updatedState().isValid()) continue;
	  TrajectoryStateOnSurface theCombinedPredictedState = 
	    TrajectoryStateCombiner().combine(itTraj->backwardPredictedState(), itTraj->forwardPredictedState());
	  TransientTrackingRecHit::ConstRecHitPointer hit = itTraj->recHit();
	  const GeomDet* det = hit->det();
			
	      // Check that the detector module belongs to the Silicon Strip detector
	      if ((det->components().empty()) &&
		  (det->subDetector() != GeomDetEnumerators::PixelBarrel) &&
		  (det->subDetector() != GeomDetEnumerators::PixelEndcap)) 
		{
		  const GeomDetUnit* du = dynamic_cast<const GeomDetUnit*>(det);
		  const Topology* theTopol = &(du->topology());
		  // residual in the measurement frame 
		  MeasurementPoint theMeasHitPos = theTopol->measurementPosition(hit->localPosition());
		  MeasurementPoint theMeasStatePos =
		    theTopol->measurementPosition( theCombinedPredictedState.localPosition());
		  Measurement2DVector residual = theMeasHitPos - theMeasStatePos;
								
		  DetId hit_detId = hit->geographicalId();				
		  int IntRawDetID = (hit_detId.rawId());
					
		  HitResidual[IntRawDetID]->Fill(residual.x()); // Fill the individual detector module Histograms
				
		  //system arranged above for the purpose of filling the histograms.
															
		}
	}
    }
}

