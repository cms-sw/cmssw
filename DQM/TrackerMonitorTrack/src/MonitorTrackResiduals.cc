// -*- C++ -*-
//
// Package:    TrackerMonitorTrack
// Class:      MonitorTrackResiduals
// 
/**\class MonitorTrackResiduals MonitorTrackResiduals.cc DQM/TrackerMonitorTrack/src/MonitorTrackResiduals.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Israel Goitom
//         Created:  Fri May 26 14:12:01 CEST 2006
// $Id: MonitorTrackResiduals.cc,v 1.21 2006/11/01 10:51:00 goitom Exp $
//
//

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

#include "Geometry/CommonDetAlgo/interface/MeasurementVector.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

#include "DataFormats/SiStripDetId/interface/TIBDetId.h"

MonitorTrackResiduals::MonitorTrackResiduals(const edm::ParameterSet& iConfig)
{
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  conf_ = iConfig;
}

MonitorTrackResiduals::~MonitorTrackResiduals()
{
}

void MonitorTrackResiduals::beginJob(edm::EventSetup const& iSetup)
{
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
  vector<uint32_t> activeDets;
  activeDets.clear(); // just in case
  tkmechstruct->addActiveDetectorsRawIds(activeDets);

  // use SiStripSubStructure for selecting certain regions
  SiStripSubStructure substructure;
  vector<uint32_t> DetIds = activeDets;
  
  vector<uint32_t> TIBDetIds;
  vector<uint32_t> TIBL1DetIds;
  vector<uint32_t> TIBL2DetIds;
  vector<uint32_t> TIBL3DetIds;
  vector<uint32_t> TIBL4DetIds;
  
  substructure.getTIBDetectors(activeDets, TIBDetIds); // this adds rawDetIds to SelectedDetIds
  
  substructure.getTIBDetectors(activeDets, TIBL1DetIds, 1); // this adds rawDetIds to SelectedDetIds
  substructure.getTIBDetectors(activeDets, TIBL2DetIds, 2); // this adds rawDetIds to SelectedDetIds
  substructure.getTIBDetectors(activeDets, TIBL3DetIds, 3); // this adds rawDetIds to SelectedDetIds
  substructure.getTIBDetectors(activeDets, TIBL4DetIds, 4); // this adds rawDetIds to SelectedDetIds
  
  vector<uint32_t> TOBDetIds;
  vector<uint32_t> TIDDetIds;
  vector<uint32_t> TECDetIds;
  substructure.getTOBDetectors(activeDets, TOBDetIds); // this adds rawDetIds to SelectedDetIds
  substructure.getTIDDetectors(activeDets, TIDDetIds); // this adds rawDetIds to SelectedDetIds
  substructure.getTECDetectors(activeDets, TECDetIds); // this adds rawDetIds to SelectedDetIds
    
    // book histo per each detector module
  int counter = 1;
  fstream file_op("IntToDetID.txt",ios::out);
  for (vector<uint32_t>::const_iterator DetItr=activeDets.begin(); DetItr!=activeDets.end(); DetItr++)
    {
      folder_organizer.setDetectorFolder(*DetItr); // pas detid - uint32 to this method - sets appropriate detector folder
      int ModuleID = (*DetItr);
      folder_organizer.setDetectorFolder(*DetItr); // top Mechanical View Folder
      string hid = hidmanager.createHistoId("HitResiduals","det",*DetItr);
      HitResidual2[ModuleID] = dbe->book1D(hid, hid, 50, -5., 5.);
	  IntToDetId[counter] = ModuleID; //Create a table of reference assigned consective number to actual module id
	  DetIdToInt[ModuleID] = counter; //create a table of reference reversing the above
	  std::cout << "Number: " << counter << " -- ModuleID: "<< ModuleID << std::endl;
	  file_op << "Number: " << counter << " -- ModuleID: "<< ModuleID << std::endl;
	  counter++;
    }
	
	file_op.close();
		  
  // book TIB histo
  vector<uint32_t>::const_iterator detid_begin = TIBDetIds.begin(); // get the first TIB Detector module
  vector<uint32_t>::const_iterator detid_end = TIBDetIds.end() -1; // get the last TIB Detector Module, -1??
  int beging=(*detid_begin); // Save first TIB Detector module id as int
  int detBegin = DetIdToInt[beging]; // Convert the detector Id to the numberrin system provided above. (i.e starting form 1)
  int ending=(*detid_end); // Save the last detector module id as int
  int detEnd = DetIdToInt[ending]; // Convert the detector Id to the numberrin system provided above. (i.e starting form 1)
  dbe->setCurrentFolder("Track/Residuals");
  HitResidual["TIB"] = dbe->bookProfile("TIBHitResiduals", "TIB Hit residuals",  TIBDetIds.size(), detBegin, detEnd, 1, -4, 4);
  
  // book TIB Layer 1 histo
  detid_begin = TIBL1DetIds.begin();
  detid_end = TIBL1DetIds.end()-1;
  beging=(*detid_begin);
  detBegin = DetIdToInt[beging];
  ending=(*detid_end);
  detEnd = DetIdToInt[ending];
  dbe->setCurrentFolder("Track/Residuals/TIB");
  HitResidual["TIBL1"] = dbe->bookProfile("TIBLayer1HitResiduals", "TIB Layer 1 Hit residuals", TIBL1DetIds.size(), detBegin, detEnd, 1, -4, 4);
 
  // book TIB Layer 2 histo
  detid_begin = TIBL2DetIds.begin();
  detid_end = TIBL2DetIds.end()-1;
  beging=(*detid_begin);
  detBegin = DetIdToInt[beging];
  ending=(*detid_end);
  detEnd = DetIdToInt[ending];
  dbe->setCurrentFolder("Track/Residuals/TIB");
  HitResidual["TIBL2"] = dbe->bookProfile("TIBLayer2HitResiduals", "TIB Layer 2 Hit residuals", TIBL2DetIds.size(), detBegin, detEnd, 1, -4, 4);


  // book TIB Layer 3 histo
  detid_begin = TIBL3DetIds.begin();
  detid_end = TIBL3DetIds.end()-1;
  beging=(*detid_begin);
  detBegin = DetIdToInt[beging];
  ending=(*detid_end);
  detEnd = DetIdToInt[ending];
  dbe->setCurrentFolder("Track/Residuals/TIB");
  HitResidual["TIBL3"] = dbe->bookProfile("TIBLayer3HitResiduals", "TIB Layer3 Hit residuals", TIBL3DetIds.size(), detBegin, detEnd, 1, -4, 4);
  
  // book TIB Layer 4 histo
  detid_begin = TIBL4DetIds.begin();
  detid_end = TIBL4DetIds.end()-1;
  beging=(*detid_begin);
  detBegin = DetIdToInt[beging];
  ending=(*detid_end);
  detEnd = DetIdToInt[ending];
  dbe->setCurrentFolder("Track/Residuals/TIB");
  HitResidual["TIBL4"] = dbe->bookProfile("TIBLayer4HitResiduals", "TIB Layer 4 Hit residuals", TIBL4DetIds.size(), detBegin, detEnd, 1, -4, 4);
  
  // book TOB histo
  detid_begin = TOBDetIds.begin();
  detid_end = TOBDetIds.end()-1;
  beging=(*detid_begin);
  detBegin = DetIdToInt[beging];
  ending=(*detid_end);
  detEnd = DetIdToInt[ending];
  dbe->setCurrentFolder("Track/Residuals");
  HitResidual["TOB"] = dbe->bookProfile("TOBHitResiduals", "TOB Hit residuals", TOBDetIds.size(), detBegin, detEnd, 1, -4, 4);

  // book TID histo
  detid_begin = TIDDetIds.begin();
  detid_end = TIDDetIds.end()-1;
  beging=(*detid_begin);
  detBegin = DetIdToInt[beging];
  ending=(*detid_end);
  detEnd = DetIdToInt[ending];
  dbe->setCurrentFolder("Track/Residuals");
  HitResidual["TID"] = dbe->bookProfile("TIDHitResiduals", "TID Hit residuals", TIDDetIds.size(), detBegin, detEnd, 1, -4, 4);

  // book TEC histo
  detid_begin = TECDetIds.begin();
  detid_end = TECDetIds.end()-1;
  beging=(*detid_begin);
  detBegin = DetIdToInt[beging];
  ending=(*detid_end);
  detEnd = DetIdToInt[ending];
  dbe->setCurrentFolder("Track/Residuals");
  HitResidual["TEC"] = dbe->bookProfile("TECHitResiduals", "TEC Hit residuals", TECDetIds.size(), detBegin, detEnd, 1, -4, 4);

}

void MonitorTrackResiduals::endJob(void)
{
  dbe->showDirStructure();
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe->save(outputFileName);
  }
}


// ------------ method called to produce the data  ------------
void MonitorTrackResiduals::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  std::string TrackCandidateProducer = conf_.getParameter<std::string>("TrackCandidateProducer");
  std::string TrackCandidateLabel = conf_.getParameter<std::string>("TrackCandidateLabel");

  ESHandle<TrackerGeometry> theRG;
  iSetup.get<TrackerDigiGeometryRecord>().get( theRG );
  
  ESHandle<MagneticField> theRMF;
  iSetup.get<IdealMagneticFieldRecord>().get( theRMF );
  
  ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  iSetup.get<TransientRecHitRecord>().get( "WithTrackAngle",theBuilder );
  
  ESHandle<TrajectoryFitter> theRFitter;
  iSetup.get<TrackingComponentsRecord>().get("KFFittingSmoother", theRFitter );
 
  const TransientTrackingRecHitBuilder* builder = theBuilder.product();
  const TrackerGeometry * theG = theRG.product();
  const MagneticField * theMF = theRMF.product();
  const TrajectoryFitter * theFitter = theRFitter.product();

  Handle<TrackCandidateCollection> trackCandidateCollection;
  iEvent.getByLabel(TrackCandidateProducer, TrackCandidateLabel, trackCandidateCollection);

  for (TrackCandidateCollection::const_iterator track = trackCandidateCollection->begin(); 
       track!=trackCandidateCollection->end(); ++track)
    {
      int test;
      std::cin >> test;
      const TrackCandidate * theTC = &(*track);
      PTrajectoryStateOnDet state = theTC->trajectoryStateOnDet();
      const TrackCandidate::range& recHitVec=theTC->recHits();
      const TrajectorySeed& seed = theTC->seed();
      std::cout<<" with "<<(int)(recHitVec.second-recHitVec.first)<<" hits"<<std::endl;

      // convert PTrajectoryStateOnDet to TrajectoryStateOnSurface
      TrajectoryStateTransform transformer;

      DetId detId(state.detId());
      TrajectoryStateOnSurface theTSOS = transformer.transientState( state, &(theG->idToDet(detId)->surface()), theMF);

      // OwnVector<TransientTrackingRecHit> hits;
      Trajectory::RecHitContainer hits;
      TrackingRecHitCollection::const_iterator hit;

      for (hit=recHitVec.first; hit!= recHitVec.second; ++hit)
	{
	  hits.push_back(builder->build(&(*hit)));
	}
	
      // do the fitting
      std::vector<Trajectory> trajVec = theFitter->fit(seed,  hits, theTSOS);
      std::cout<<"Fitted candidate with "<<trajVec.size()<<" tracks"<<std::endl;

      if (trajVec.size() != 0)
	{
	  const Trajectory& theTraj = trajVec.front();
		
	  Trajectory::DataContainer fits = theTraj.measurements();
	  for (Trajectory::DataContainer::iterator fit=fits.begin(); fit != fits.end(); fit++)
	    {
	      const TrajectoryMeasurement tm = *fit;
	      TrajectoryStateOnSurface theCombinedPredictedState = 
		TrajectoryStateCombiner().combine( tm.forwardPredictedState(), tm.backwardPredictedState());
	      TransientTrackingRecHit::ConstRecHitPointer hit = tm.recHit();
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
					
		  HitResidual2[IntRawDetID]->Fill(residual.x()); // Fill the individual detector module Histograms
				
		  int IntRawDetID2 = DetIdToInt[IntRawDetID]; // Convert the detector module id to the numbering 
		  //system arranged above for the purpose of filling the histograms.
															
		  TIBDetId layer(hit_detId);
				
		  switch(hit_detId.subdetId()) // Check which sub-detector section a module belongs to
		    {
		    case StripSubdetector::TIB :
		      HitResidual["TIB"]->Fill(IntRawDetID2, residual.x());

		      std::cout << "\n\n\n\n\n *** Filling Histos **** \n\n\n\n" << endl;

		      if (layer.layer()==1) HitResidual["TIBL1"]->Fill(IntRawDetID2, residual.x());
		      if (layer.layer()==2) HitResidual["TIBL2"]->Fill(IntRawDetID2, residual.x());
		      if (layer.layer()==3) HitResidual["TIBL3"]->Fill(IntRawDetID2, residual.x());
		      if (layer.layer()==4) HitResidual["TIBL4"]->Fill(IntRawDetID2, residual.x());
		      break;
		    case StripSubdetector::TOB :
		      HitResidual["TOB"]->Fill(IntRawDetID2, residual.x());
		      break;
		    case StripSubdetector::TID :
		      HitResidual["TID"]->Fill(IntRawDetID2, residual.x());
		      break;
		    case StripSubdetector::TEC :
		      HitResidual["TEC"]->Fill(IntRawDetID2, residual.x());
		      break;
		    default:
		      break;
		    }
		}
	    }
	}
    }
}

//define this as a plug-in
//DEFINE_FWK_MODULE(MonitorTrackResiduals)
