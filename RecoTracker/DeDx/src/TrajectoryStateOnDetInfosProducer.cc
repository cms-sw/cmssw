// -*- C++ -*-
//
// Package:    TrajectoryStateOnDetInfosProducer
// Class:      TrajectoryStateOnDetInfosProducer
// 
/**\class TrajectoryStateOnDetInfosProducer TrajectoryStateOnDetInfosProducer.cc RecoTracker/TrajectoryStateOnDetInfosProducer/src/TrajectoryStateOnDetInfosProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  loic Quertenmont (querten)
//         Created:  Thu May 10 14:09:02 CEST 2008
// $Id: TrajectoryStateOnDetInfosProducer.cc,v 1.2 2008/05/10 18:57:23 querten Exp $
//
//


// system include files
#include <memory>
#include <vector>
#include <map>

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackReco/interface/TrajectoryStateOnDetInfo.h"
#include "DataFormats/TrackReco/interface/TrackTrajectoryStateOnDetInfos.h"
#include "RecoTracker/DeDx/interface/TrajectoryStateOnDetInfosProducer.h"
#include "RecoTracker/DeDx/interface/TrajectoryStateOnDetInfosTools.h"


using namespace reco;
using namespace std;
using namespace edm;


TrajectoryStateOnDetInfosProducer::TrajectoryStateOnDetInfosProducer(const edm::ParameterSet& iConfig)
{
   m_trajTrackAssociationTag   = iConfig.getParameter<edm::InputTag>("trajectoryTrackAssociation");
   m_tracksTag                 = iConfig.getParameter<edm::InputTag>("Track");

   Track_PMin                  = iConfig.getUntrackedParameter<double>("Track_PMin"    ,0);
   Track_PMax                  = iConfig.getUntrackedParameter<double>("Track_PMax"    ,9999999);
   Track_Chi2Max               = iConfig.getUntrackedParameter<double>("Track_Chi2Max" ,9999999);


   produces<reco::TrackTrajectoryStateOnDetInfosCollection>();  
}


TrajectoryStateOnDetInfosProducer::~TrajectoryStateOnDetInfosProducer()
{ 
}

void
TrajectoryStateOnDetInfosProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
   iEvent.getByLabel(m_trajTrackAssociationTag, trajTrackAssociationHandle);
   const TrajTrackAssociationCollection TrajToTrackMap = *trajTrackAssociationHandle.product();

   edm::Handle<reco::TrackCollection> trackCollectionHandle;
   iEvent.getByLabel(m_tracksTag,trackCollectionHandle);


   TrackTrajectoryStateOnDetInfosCollection* outputCollection = new TrackTrajectoryStateOnDetInfosCollection(reco::TrackRefProd(trackCollectionHandle) );
   TSODI::Fill_TSODICollection(TrajToTrackMap, outputCollection);
   
   //put in the event the result
   std::auto_ptr<TrackTrajectoryStateOnDetInfosCollection> outputs(outputCollection);
   iEvent.put(outputs);
}

void 
TrajectoryStateOnDetInfosProducer::beginJob(const edm::EventSetup&)
{
}

void 
TrajectoryStateOnDetInfosProducer::endJob() {
}

DEFINE_FWK_MODULE(TrajectoryStateOnDetInfosProducer);
