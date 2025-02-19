#ifndef PFDisplacedTrackerVertexProducer_H
#define PFDisplacedTrackerVertexProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

class PFTrackTransformer;
class PFDisplacedTrackerVertexProducer : public edm::EDProducer {
public:
  
  ///Constructor
  explicit PFDisplacedTrackerVertexProducer(const edm::ParameterSet&);
  
  ///Destructor
  ~PFDisplacedTrackerVertexProducer();
  
private:
  virtual void beginRun(edm::Run&,const edm::EventSetup&) ;
  virtual void endRun() ;
  
  ///Produce the PFRecTrack collection
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
  ///PFTrackTransformer
  PFTrackTransformer *pfTransformer_; 
  edm::InputTag pfDisplacedVertexContainer_;
  edm::InputTag pfTrackContainer_;

};
#endif
