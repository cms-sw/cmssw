#ifndef TrajectorySateOnDetInfosProducer_H
#define TrajectorySateOnDetInfosProducer_H
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackReco/interface/Track.h"

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
#include "DataFormats/TrackReco/interface/TrackTrajectorySateOnDetInfos.h"
#include "DataFormats/TrackReco/interface/TrajectorySateOnDetInfo.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

//
// class decleration
//
#include <map>

using namespace reco;


class TrackerGeometry;

class TrajectorySateOnDetInfosProducer : public edm::EDProducer {
   public:
      explicit TrajectorySateOnDetInfosProducer(const edm::ParameterSet&);
      ~TrajectorySateOnDetInfosProducer();

       TrackTrajectorySateOnDetInfosCollection* Get_TSODICollection(edm::Event& iEvent, const edm::EventSetup& iSetup);

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      reco::TrajectorySateOnDetInfo* Get_TSODI(const Trajectory*, const TrajectoryStateOnSurface*, const SiStripRecHit2D*);

     
      // ----------member data ---------------------------
     edm::InputTag m_trajTrackAssociationTag;
     edm::InputTag m_tracksTag;

     double Track_PMin;
     double Track_PMax;
     double Track_Chi2Max;
     
};

#endif

