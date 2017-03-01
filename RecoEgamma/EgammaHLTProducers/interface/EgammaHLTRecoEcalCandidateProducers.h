// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTRecoEcalCandidateProducers
// 
/**\class EgammaHLTRecoEcalCandidateProducers.h EgammaHLTRecoEcalCandidateProducers.cc  RecoEgamma/EgammaHLTProducers/interface/EgammaHLTRecoEcalCandidateProducers.h.h
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id:
//
//

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTRecoEcalCandidateProducers : public edm::global::EDProducer<> {

 public:

  EgammaHLTRecoEcalCandidateProducers (const edm::ParameterSet& ps);
  ~EgammaHLTRecoEcalCandidateProducers();

  void produce(edm::StreamID sid, edm::Event& evt, const edm::EventSetup& es) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  const edm::EDGetTokenT<reco::SuperClusterCollection> scHybridBarrelProducer_;
  const edm::EDGetTokenT<reco::SuperClusterCollection> scIslandEndcapProducer_;
  const std::string recoEcalCandidateCollection_;
};


