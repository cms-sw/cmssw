#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/TrackerMonitorTrack/interface/MonitorTrackResiduals.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "DQMServices/Core/interface/DQMStore.h"

MonitorTrackResiduals::MonitorTrackResiduals(const edm::ParameterSet& iConfig) {
  dqmStore_ = edm::Service<DQMStore>().operator->();
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
      HitResidual[ModuleID] = dqmStore_->book1D(hid, hid, 50, -5., 5.);
      HitResidual[ModuleID]->setAxisTitle("Hit residuals on tracks crossing this detector module");
    }
	

}

void MonitorTrackResiduals::endJob(void) {
  dqmStore_->showDirStructure();
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dqmStore_->save(outputFileName);
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
DEFINE_FWK_MODULE(MonitorTrackResiduals);

