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
// $Id: DeDxEstimatorProducer.cc,v 1.5 2007/06/13 12:04:26 arizzi Exp $
//
//


// system include files
#include <memory>

#include "RecoTracker/DeDx/interface/DeDxEstimatorProducer.h"
#include "DataFormats/TrackReco/interface/TrackDeDxEstimate.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "RecoTracker/DeDx/interface/GenericAverageDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/TruncatedAverageDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/MedianDeDxEstimator.h"
using namespace reco;
using namespace std;
//
// constructors and destructor
//
DeDxEstimatorProducer::DeDxEstimatorProducer(const edm::ParameterSet& iConfig)
{
   //register your products
   produces<TrackDeDxEstimateCollection>();
   m_trackDeDxHitsTag = iConfig.getParameter<edm::InputTag>("trackDeDxHits");

   //FIXME: configurable, use ES?
   string estimatorName = iConfig.getParameter<string>("estimator");
   if(estimatorName == "median")  m_estimator = new MedianDeDxEstimator(-2.);
   if(estimatorName == "generic")  m_estimator = new GenericAverageDeDxEstimator(iConfig.getParameter<double>("exponent"));
   if(estimatorName == "truncated")  m_estimator = new TruncatedAverageDeDxEstimator(iConfig.getParameter<double>("fraction"));

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
   edm::Handle<reco::TrackDeDxHitsCollection> trackDeDxHitsCollectionHandle;
   iEvent.getByLabel(m_trackDeDxHitsTag,trackDeDxHitsCollectionHandle);
   const reco::TrackDeDxHitsCollection & hits = *trackDeDxHitsCollectionHandle.product();
   TrackDeDxEstimateCollection * outputCollection = new TrackDeDxEstimateCollection(hits.keyProduct());
   
   reco::TrackDeDxHitsCollection::const_iterator it= hits.begin();
   for(int j=0;it!=hits.end();++it,j++)
   {
      //FIXME: insert here some code to suppress pixel usage if wanted 
      float val=m_estimator->dedx(*it);
      outputCollection->setValue(j, val);
   }
   
   //put in the event the result
    std::auto_ptr<TrackDeDxEstimateCollection> estimator(outputCollection);
    iEvent.put(estimator);
   
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
