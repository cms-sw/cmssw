#ifndef SHALLOW_TRACKS_PRODUCER
#define SHALLOW_TRACKS_PRODUCER

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"

class ShallowTracksProducer : public edm::EDProducer {
public:
  explicit ShallowTracksProducer(const edm::ParameterSet&);
private:
  edm::InputTag theTracksLabel;
  std::string Prefix;
  std::string Suffix;
  void produce( edm::Event &, const edm::EventSetup & );
};
#endif
