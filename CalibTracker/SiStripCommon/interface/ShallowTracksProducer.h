#ifndef SHALLOW_TRACKS_PRODUCER
#define SHALLOW_TRACKS_PRODUCER

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"

class ShallowTracksProducer : public edm::EDProducer {
public:
  explicit ShallowTracksProducer(const edm::ParameterSet&);
private:
	const edm::EDGetTokenT<edm::View<reco::Track> > tracks_token_;
  edm::InputTag theTracksLabel;
  std::string Prefix;
  std::string Suffix;
  void produce( edm::Event &, const edm::EventSetup & ) override;
};
#endif
