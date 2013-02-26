#ifndef PFTrackProducer_H
#define PFTrackProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
/// \brief Abstract
/*!
\author Daniele Benedetti
\date November 2010
  PFTrackTransformer transforms all the tracks in the PFRecTracks.
  NOTE the PFRecTrack collection is transient.  
*/


class PFTrackTransformer;
class PFTrackProducer : public edm::EDProducer {
public:
  
  ///Constructor
  explicit PFTrackProducer(const edm::ParameterSet&);
  
  ///Destructor
  ~PFTrackProducer();
  
private:
  virtual void beginRun(const edm::Run&,const edm::EventSetup&) override;
  virtual void endRun(const edm::Run&,const edm::EventSetup&) override;
  
  ///Produce the PFRecTrack collection
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  
  ///PFTrackTransformer
  PFTrackTransformer *pfTransformer_; 
  std::vector<edm::InputTag> tracksContainers_;
  edm::InputTag gsfTrackLabel_;  
  edm::InputTag muonColl_;
  edm::InputTag vtx_h;
  ///TRACK QUALITY
  bool useQuality_;
  reco::TrackBase::TrackQuality trackQuality_;
  bool trajinev_;
  bool gsfinev_;

};
#endif
