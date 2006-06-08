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
// $Id: MonitorTrackResiduals.cc,v 1.5 2006/06/04 17:52:23 goitom Exp $
//
//


// system include files

// user include files
#include "DQM/TrackerMonitorTrack/interface/MonitorTrackResiduals.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoTracker/TrackProducer/interface/TrackingRecHitLessFromGlobalPosition.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
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
  using namespace std;

  dbe->setCurrentFolder("SiStrip/Detectors");

  // take from eventSetup the SiStripStructure object - here will use SiStripDetControl later on

    edm::ESHandle<SiStripDetCabling> tkmechstruct;
    iSetup.get<SiStripDetCablingRcd>().get(tkmechstruct);

  // get list of active detectors from SiStripStructure - this will change and be taken from a SiStripDetControl constobject 

    vector<uint32_t> activeDets;
    activeDets.clear(); // just in case
    tkmechstruct->addActiveDetectorsRawIds(activeDets);

  // use SiStripSubStructure for selecting certain regions
  SiStripSubStructure substructure;

  vector<uint32_t> DetIds = activeDets;
  vector<uint32_t> TIBDetIds;
  vector<uint32_t> TOBDetIds;
  vector<uint32_t> TIDDetIds;
  vector<uint32_t> TECDetIds;

  substructure.getTIBDetectors(activeDets, TIBDetIds); // this adds rawDetIds to SelectedDetIds
  substructure.getTOBDetectors(activeDets, TOBDetIds);    // this adds rawDetIds to SelectedDetIds
  substructure.getTIDDetectors(activeDets, TIDDetIds); // this adds rawDetIds to SelectedDetIds
  substructure.getTECDetectors(activeDets, TECDetIds); // this adds rawDetIds to SelectedDetIds

  
  vector<uint32_t>::const_iterator detid_begin = TIBDetIds.begin();
  vector<uint32_t>::const_iterator detid_end = TIBDetIds.end() -1;


  int detBegin=(*detid_begin)&0x1ffffff;
  int detEnd=(*detid_end)&0x1ffffff;

  HitResidual["TIB"] = dbe->bookProfile("TIBHitRes", "TIB Hit residuals",  TIBDetIds.size(), detBegin, detEnd, 1, -4, 4);

  detid_begin = TOBDetIds.begin();
  detid_end = TOBDetIds.end()-1;

  detBegin=(*detid_begin)&0x1ffffff;
  detEnd=(*detid_end)&0x1ffffff;

  HitResidual["TOB"] = dbe->bookProfile("TOBHitRes", "TOB Hit residuals", TOBDetIds.size(), detBegin, detEnd, 1, -4, 4);

  detid_begin = TIDDetIds.begin();
  detid_end = TIDDetIds.end()-1;

  detBegin=(*detid_begin)&0x1ffffff;
  detEnd=(*detid_end)&0x1ffffff;

  HitResidual["TID"] = dbe->bookProfile("TIDHitRes", "TID Hit residuals", TIDDetIds.size(), detBegin, detEnd, 1, -4, 4);

  detid_begin = TECDetIds.begin();
  detid_end = TECDetIds.end()-1;

  detBegin=(*detid_begin)&0x1ffffff;
  detEnd=(*detid_end)&0x1ffffff;

  HitResidual["TEC"] = dbe->bookProfile("TECHitRes", "TEC Hit residuals", TECDetIds.size(), detBegin, detEnd, 1, -4, 4);

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

//
// member functions
//

// ------------ method called to produce the data  ------------
void
MonitorTrackResiduals::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif

   std::string TrackCandidateProducer = conf_.getParameter<std::string>("TrackCandidateProducer");
   std::string TrackCandidateLabel = conf_.getParameter<std::string>("TrackCandidateLabel");

   ESHandle<TrackerGeometry> theRG;
   ESHandle<MagneticField> theRMF;
   edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
   edm::ESHandle<TrajectoryFitter> theRFitter;

   const TransientTrackingRecHitBuilder* builder = theBuilder.product();
   const TrackingGeometry * theG = theRG.product();
   const MagneticField * theMF = theRMF.product();
   const TrajectoryFitter * theFitter = theRFitter.product();

   Handle<TrackCandidateCollection> trackCandidateCollection;
   iEvent.getByLabel(TrackCandidateProducer, TrackCandidateLabel, trackCandidateCollection);

   ESHandle<TrackerGeometry> pDD;
   iSetup.get<TrackerDigiGeometryRecord>().get( pDD );

    edm::ESHandle<SiStripDetCabling> tkmechstruct;
    iSetup.get<SiStripDetCablingRcd>().get(tkmechstruct);

  // get list of active detectors from SiStripStructure - this will change and be taken from a SiStripDetControl constobject 

    std::vector<uint32_t> activeDets;
    activeDets.clear(); // just in case
    tkmechstruct->addActiveDetectorsRawIds(activeDets);

  // use SiStripSubStructure for selecting certain regions
  SiStripSubStructure substructure;

  std::vector<uint32_t> TIBDetIds;
  std::vector<uint32_t> TOBDetIds;
  std::vector<uint32_t> TIDDetIds;
  std::vector<uint32_t> TECDetIds;

  substructure.getTIBDetectors(activeDets, TIBDetIds); // this adds rawDetIds to SelectedDetIds
  substructure.getTOBDetectors(activeDets, TOBDetIds);    // this adds rawDetIds to SelectedDetIds
  substructure.getTIDDetectors(activeDets, TIDDetIds); // this adds rawDetIds to SelectedDetIds
  substructure.getTECDetectors(activeDets, TECDetIds); // this adds rawDetIds to SelectedDetIds

  
  std::vector<uint32_t>::const_iterator TIBdetid_begin = TIBDetIds.begin();
  std::vector<uint32_t>::const_iterator TIBdetid_end = TIBDetIds.end() -1;
  std::vector<uint32_t>::const_iterator TOBdetid_begin = TOBDetIds.begin();
  std::vector<uint32_t>::const_iterator TOBdetid_end = TOBDetIds.end() -1;
  std::vector<uint32_t>::const_iterator TIDdetid_begin = TIDDetIds.begin();
  std::vector<uint32_t>::const_iterator TIDdetid_end = TIDDetIds.end() -1;
  std::vector<uint32_t>::const_iterator TECdetid_begin = TECDetIds.begin();
  std::vector<uint32_t>::const_iterator TECdetid_end = TECDetIds.end() -1;

  for (TrackCandidateCollection::const_iterator track = trackCandidateCollection->begin(); track!=trackCandidateCollection->end(); ++track)
    {

      const TrackCandidate * theTC = &(*track);
      PTrajectoryStateOnDet state = theTC->trajectoryStateOnDet();
      const TrackCandidate::range& recHitVec=theTC->recHits();
      const TrajectorySeed& seed = theTC->seed();

      //convert PTrajectoryStateOnDet to TrajectoryStateOnSurface
      TrajectoryStateTransform transformer;
  
      DetId * detId = new DetId(state.detId());
      TrajectoryStateOnSurface theTSOS = transformer.transientState( state, &(theG->idToDet(*detId)->surface()), theMF);

      OwnVector<TransientTrackingRecHit> hits;
      float ndof=0;

      TrackingRecHitCollection::const_iterator hit;
      for (hit=recHitVec.first; hit!= recHitVec.second; ++hit)
	{
	  hits.push_back(builder->build(&(*hit)));
	  if ((*hit).isValid())
	    {
	      ndof = ndof + (hit->dimension())*(hit->weight());
	    }

	  ndof = ndof - 5;

	  std::vector<Trajectory> trajVec;
	  //reco::Track * theTrack;
	  Trajectory * theTraj;

	  trajVec = theFitter->fit(seed,  hits, theTSOS);

	  TrajectoryStateOnSurface innertsos;  
	  if (trajVec.size() != 0)
	    {   
	      theTraj = new Trajectory( trajVec.front() );
    
	      if (theTraj->direction() == alongMomentum)
		{      
		  innertsos = theTraj->firstMeasurement().updatedState();    
		} else 
		{
		  innertsos = theTraj->lastMeasurement().updatedState();    
		}
	    }

	  /*
	  std::vector<TrajectoryMeasurement> measurements = theTraj->measurements();
	  for (std::vector<TrajectoryMeasurement>::const_iterator i = measurements.begin(); i != measurements.end(); ++i)
	    {
	      i->predictedState().localPosition().x;
	    }
	  */

	  //double hitPos = hit->localPosition().x();
	  DetId detId = hit->geographicalId();
	  //const GeomDetUnit *detUnit = pDD->idToDetUnit(detId);
	  //const BoundPlane & localSurface = detUnit->surface();

	  double residual = innertsos.localPosition().x() - hit->localPosition().x();

	  uint32_t rawDetId = detId.rawId();
          int CutRawDetId=(rawDetId)&0x1ffffff;
	  if (rawDetId>=(*TIBdetid_begin) || rawDetId<=(*TIBdetid_end))
	    {
	      HitResidual["TIB"]->Fill(CutRawDetId, residual);
	    }
	  if (rawDetId>=(*TOBdetid_begin) || rawDetId<=(*TOBdetid_end))
	    {
	      HitResidual["TOB"]->Fill(CutRawDetId, residual);
	    }
	  if (rawDetId>=(*TIDdetid_begin) || rawDetId<=(*TIDdetid_end))
	    {
	      HitResidual["TID"]->Fill(CutRawDetId, residual);
	    }
	  if (rawDetId>=(*TECdetid_begin) || rawDetId<=(*TECdetid_end))
	    {
	      HitResidual["TEC"]->Fill(CutRawDetId, residual);
	    }
	}
    }
}

//define this as a plug-in
//DEFINE_FWK_MODULE(MonitorTrackResiduals)
