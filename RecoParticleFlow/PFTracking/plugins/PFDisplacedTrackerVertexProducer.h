#ifndef PFDisplacedTrackerVertexProducer_H
#define PFDisplacedTrackerVertexProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedTrackerVertex.h"

class PFTrackTransformer;
class PFDisplacedTrackerVertexProducer : public edm::stream::EDProducer<> {
public:
  
  ///Constructor
  explicit PFDisplacedTrackerVertexProducer(const edm::ParameterSet&);
  
  ///Destructor
  ~PFDisplacedTrackerVertexProducer();
  
private:
  virtual void beginRun(const edm::Run&,const edm::EventSetup&) override;
  virtual void endRun(const edm::Run&,const edm::EventSetup&) override;
  
  ///Produce the PFRecTrack collection
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  
  ///PFTrackTransformer
  PFTrackTransformer *pfTransformer_; 
  edm::EDGetTokenT<reco::PFDisplacedVertexCollection> pfDisplacedVertexContainer_;
  edm::EDGetTokenT<reco::TrackCollection> pfTrackContainer_;

};
#endif
