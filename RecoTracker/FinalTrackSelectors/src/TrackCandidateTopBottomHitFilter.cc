// -*- C++ -*-
//
// Package:    TrackCandidateTopBottomHitFilter
// Class:      TrackCandidateTopBottomHitFilter
// 
/**\class TrackCandidateTopBottomHitFilter TrackCandidateTopBottomHitFilter.cc Cruzet/TrackCandidateTopBottomHitFilter/src/TrackCandidateTopBottomHitFilter.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Giuseppe Cerati
//         Created:  Tue Sep  9 09:47:01 CEST 2008
// $Id: TrackCandidateTopBottomHitFilter.cc,v 1.4 2013/02/27 14:58:17 muzaffar Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 

//
// class decleration
//

class TrackCandidateTopBottomHitFilter : public edm::EDProducer {
public:
  explicit TrackCandidateTopBottomHitFilter(const edm::ParameterSet&);
  ~TrackCandidateTopBottomHitFilter();

private:
  virtual void beginRun(edm::Run const& run, const edm::EventSetup&) override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() ;
  edm::InputTag label;
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  std::string builderName;
  double seedY;
};

TrackCandidateTopBottomHitFilter::TrackCandidateTopBottomHitFilter(const edm::ParameterSet& iConfig) {
  builderName = iConfig.getParameter<std::string>("TTRHBuilder");   
  label = iConfig.getParameter<edm::InputTag>("Input");
  seedY = iConfig.getParameter<double>("SeedY");

  produces<TrackCandidateCollection>();  
}


TrackCandidateTopBottomHitFilter::~TrackCandidateTopBottomHitFilter() {}


//
// member functions
//

// ------------ method called to produce the data  ------------
void TrackCandidateTopBottomHitFilter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  Handle<TrackCandidateCollection> pIn;
  iEvent.getByLabel(label,pIn);
  std::auto_ptr<TrackCandidateCollection> pOut(new TrackCandidateCollection);
  for (TrackCandidateCollection::const_iterator it=pIn->begin(); it!=pIn->end();++it) {
    PTrajectoryStateOnDet state = it->trajectoryStateOnDet();   
    TrackCandidate::range oldhits = it->recHits();
    TrajectorySeed seed = it->seed();
    TrackCandidate::RecHitContainer hits;
    for (TrackCandidate::RecHitContainer::const_iterator hit=oldhits.first;hit!=oldhits.second;++hit) {
      if (hit->isValid()) {
	double hitY = theBuilder->build(&*hit)->globalPosition().y();
	if (seedY*hitY>0) hits.push_back(hit->clone());
	else break;
      } else hits.push_back(hit->clone());
    }
    if (hits.size()>=3) {
      TrackCandidate newTC(hits,seed,state);
      pOut->push_back(newTC);
    }
  }
  iEvent.put(pOut);
}

void TrackCandidateTopBottomHitFilter::beginRun(edm::Run const& run, const edm::EventSetup& iSetup) {
  iSetup.get<TransientRecHitRecord>().get(builderName,theBuilder);
}

void TrackCandidateTopBottomHitFilter::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackCandidateTopBottomHitFilter);
