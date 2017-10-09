#ifndef PFV0Producer_H
#define PFV0Producer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0Fwd.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"


class PFTrackTransformer;
class PFV0Producer : public edm::stream::EDProducer<> {
public:
  
  ///Constructor
  explicit PFV0Producer(const edm::ParameterSet&);
  
  ///Destructor
  ~PFV0Producer();
  
private:
  virtual void beginRun(const edm::Run&,const edm::EventSetup&) override;
  virtual void endRun(const edm::Run&,const edm::EventSetup&) override;
  
  ///Produce the PFRecTrack collection
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  ///PFTrackTransformer
  PFTrackTransformer *pfTransformer_; 
  std::vector < edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> > V0list_;

};
#endif
