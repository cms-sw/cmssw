#ifndef PFV0Producer_H
#define PFV0Producer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"


class PFTrackTransformer;
class PFV0Producer : public edm::EDProducer {
public:
  
  ///Constructor
  explicit PFV0Producer(const edm::ParameterSet&);
  
  ///Destructor
  ~PFV0Producer();
  
private:
  virtual void beginRun(edm::Run&,const edm::EventSetup&) ;
  virtual void endRun() ;
  
  ///Produce the PFRecTrack collection
  virtual void produce(edm::Event&, const edm::EventSetup&);

  ///PFTrackTransformer
  PFTrackTransformer *pfTransformer_; 
  std::vector < edm::InputTag > V0list_;

};
#endif
