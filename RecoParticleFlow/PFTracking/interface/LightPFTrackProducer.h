#ifndef LightPFTrackProducer_H
#define LightPFTrackProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

class PFTrackTransformer;
class LightPFTrackProducer : public edm::EDProducer {
public:
  
  ///Constructor
  explicit LightPFTrackProducer(const edm::ParameterSet&);
  
  ///Destructor
  ~LightPFTrackProducer();
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;
  
  ///Produce the PFRecTrack collection
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
  ///PFTrackTransformer
  PFTrackTransformer *pfTransformer_; 
  std::vector<edm::InputTag> tracksContainers_;
};
#endif
