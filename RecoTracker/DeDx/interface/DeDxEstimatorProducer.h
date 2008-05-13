#ifndef DeDxEstimatorProducer_H
#define DeDxEstimatorProducer_H
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/DeDx/interface/BaseDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/TrajectorySateOnDetInfosProducer.h"


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

      // ----------member data ---------------------------
      BaseDeDxEstimator *               m_estimator;
      edm::InputTag                     m_TsodiTag;
      bool                              m_FromTrajectory;
      TrajectorySateOnDetInfosProducer* m_TSODIProducer;
};

#endif

