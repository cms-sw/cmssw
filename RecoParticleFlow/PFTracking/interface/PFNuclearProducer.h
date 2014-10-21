#ifndef PFNuclearProducer_H
#define PFNuclearProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFNuclearInteraction.h"

class PFTrackTransformer;
class PFNuclearProducer : public edm::stream::EDProducer<> {
public:
  
  ///Constructor
  explicit PFNuclearProducer(const edm::ParameterSet&);
  
  ///Destructor
  ~PFNuclearProducer();
  
private:
  virtual void beginRun(const edm::Run&,const edm::EventSetup&) override;
  virtual void endRun(const edm::Run&,const edm::EventSetup&) override;
  
  ///Produce the PFRecTrack collection
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  
  ///PFTrackTransformer
  PFTrackTransformer *pfTransformer_; 
  double likelihoodCut_;
  std::vector<edm::EDGetTokenT<reco::NuclearInteractionCollection> > nuclearContainers_;
};
#endif
