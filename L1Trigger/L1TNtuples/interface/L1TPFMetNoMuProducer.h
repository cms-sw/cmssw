#ifndef L1TPFMetNoMuProducer_H
#define L1TPFMetNoMuProducer_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

class L1TPFMetNoMuProducer : public edm::stream::EDProducer<> {
public:
  explicit L1TPFMetNoMuProducer(const edm::ParameterSet &ps);


private:
  void produce(edm::Event &event, const edm::EventSetup &eventSetup);

  const edm::EDGetTokenT<reco::PFMETCollection> thePFMETCollection_;
  const edm::EDGetTokenT<reco::MuonCollection> theMuonCollection_;
};
#endif
