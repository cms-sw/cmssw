#ifndef MuonIdentification_MuonLinksProducer_h
#define MuonIdentification_MuonLinksProducer_h

// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonLinksProducer
//
/*
 Simple producer to make reco::MuonTrackLinks collection 
 out of the global muons from "muons" collection to restore
 dropped links used as input for MuonIdProducer.
 */
//
// Original Author:  Dmytro Kovalskyi
//
//

// user include files
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class MuonLinksProducer : public edm::global::EDProducer<> {
public:
  explicit MuonLinksProducer(const edm::ParameterSet&);
  ~MuonLinksProducer() override = default;

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::InputTag m_inputCollection;
  const edm::EDGetTokenT<reco::MuonCollection> muonToken_;
};
#endif
