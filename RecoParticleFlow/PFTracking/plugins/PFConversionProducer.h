#ifndef PFConversionProducer_H
#define PFConversionProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

class PFTrackTransformer;
class PFConversionProducer : public edm::EDProducer {
public:
  
  ///Constructor
  explicit PFConversionProducer(const edm::ParameterSet&);
  
  ///Destructor
  ~PFConversionProducer();
  
private:
  virtual void beginRun(edm::Run&,const edm::EventSetup&) ;
  virtual void endRun() ;
  
  ///Produce the PFRecTrack collection
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
  ///PFTrackTransformer
  PFTrackTransformer *pfTransformer_; 
  edm::InputTag pfConversionContainer_;
  edm::InputTag pfTrackContainer_;
  edm::InputTag vtx_h;
};
#endif
