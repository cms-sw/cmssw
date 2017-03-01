#ifndef LightPFTrackProducer_H
#define LightPFTrackProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class PFTrackTransformer;
class LightPFTrackProducer : public edm::stream::EDProducer<> {
public:
  
  ///Constructor
  explicit LightPFTrackProducer(const edm::ParameterSet&);
  
  ///Destructor
  ~LightPFTrackProducer();
  
private:
  virtual void beginRun(const edm::Run&,const edm::EventSetup&) override ;
  virtual void endRun(const edm::Run&,const edm::EventSetup&) override;
  
  ///Produce the PFRecTrack collection
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  
  ///PFTrackTransformer
  PFTrackTransformer *pfTransformer_; 
  std::vector<edm::EDGetTokenT<reco::TrackCollection> > tracksContainers_;
  ///TRACK QUALITY
    bool useQuality_;
    reco::TrackBase::TrackQuality trackQuality_;

};
#endif
