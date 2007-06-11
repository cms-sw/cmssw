// -*- C++ -*-
//
// Package:    DeDxEstimatorProducer
// Class:      DeDxEstimatorProducer
// 
/**\class DeDxEstimatorProducer DeDxEstimatorProducer.cc RecoTracker/DeDxEstimatorProducer/src/DeDxEstimatorProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  andrea
//         Created:  Thu May 31 14:09:02 CEST 2007
// $Id$
//
//


// system include files
#include <memory>

#include "RecoTracker/DeDx/interface/DeDxEstimatorProducer.h"
#include "RecoTracker/DeDx/interface/GenericAverageDeDxEstimator.h"
#include "DataFormats/TrackReco/interface/TrackDeDxEstimator.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"

using namespace reco;
//
// constructors and destructor
//
DeDxEstimatorProducer::DeDxEstimatorProducer(const edm::ParameterSet& iConfig)
{
   //register your products
   produces<TrackDeDxEstimatorCollection>();

   //FIXME: configurable
   m_estimator = new GenericAverageDeDxEstimator(-2.);

}


DeDxEstimatorProducer::~DeDxEstimatorProducer()
{
 //FIXME: only if doen't come from ES
 delete m_estimator;
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DeDxEstimatorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
//TODO: loop on tracks+dedxhits and apply the estimator

}

// ------------ method called once each job just before starting event loop  ------------
void 
DeDxEstimatorProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DeDxEstimatorProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeDxEstimatorProducer);
