#ifndef PFConversionProducer_H
#define PFConversionProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"

class PFTrackTransformer;
class PFConversionProducer : public edm::stream::EDProducer<> {
public:
  
  ///Constructor
  explicit PFConversionProducer(const edm::ParameterSet&);
  
  ///Destructor
  ~PFConversionProducer();
  
private:
  virtual void beginRun(const edm::Run&,const edm::EventSetup&) override;
  virtual void endRun(const edm::Run&,const edm::EventSetup&) override;
  
  ///Produce the PFRecTrack collection
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  
  ///PFTrackTransformer
  PFTrackTransformer *pfTransformer_; 
  edm::EDGetTokenT<reco::ConversionCollection> pfConversionContainer_;
  edm::EDGetTokenT<reco::VertexCollection> vtx_h;
  
};
#endif
