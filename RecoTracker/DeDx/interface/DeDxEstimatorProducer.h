#ifndef DeDxEstimatorProducer_H
#define DeDxEstimatorProducer_H
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/DeDx/interface/BaseDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/TrajectoryStateOnDetInfosProducer.h"


//
// class decleration
//

class DeDxEstimatorProducer : public edm::EDProducer {
   public:
      explicit DeDxEstimatorProducer(const edm::ParameterSet&);
      ~DeDxEstimatorProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      std::vector<Measurement1D> GetMeasurements(TrajectoryStateOnDetInfoCollection Tsodis, edm::ESHandle<TrackerGeometry> tkGeom);

      // ----------member data ---------------------------
      BaseDeDxEstimator*                m_estimator;
      edm::InputTag                     m_TsodiTag;
      bool                              m_FromTrajectory;
      edm::InputTag                     m_trajTrackAssociationTag;
      edm::InputTag                     m_tracksTag;

};

#endif

