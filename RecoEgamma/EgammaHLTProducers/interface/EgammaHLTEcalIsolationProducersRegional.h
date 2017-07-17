// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTEcalIsolationProducersRegional
// 
/**\class EgammaHLTEcalIsolationProducersRegional EgammaHLTEcalIsolationProducersRegional.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTEcalIsolationProducersRegional.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTEcalIsolationProducersRegional.h,v 1.2 2008/05/12 08:48:22 ghezzi Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTEcalIsolation.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTEcalIsolationProducersRegional : public edm::EDProducer {
public:
  explicit EgammaHLTEcalIsolationProducersRegional(const edm::ParameterSet&);
  ~EgammaHLTEcalIsolationProducersRegional();
  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  edm::EDGetTokenT<reco::BasicClusterCollection> bcBarrelProducer_;
  edm::EDGetTokenT<reco::BasicClusterCollection> bcEndcapProducer_;
  edm::EDGetTokenT<reco::SuperClusterCollection> scIslandBarrelProducer_;
  edm::EDGetTokenT<reco::SuperClusterCollection> scIslandEndcapProducer_;

  edm::ParameterSet conf_;

  double  egEcalIsoEtMin_;
  double  egEcalIsoConeSize_;
  int algoType_;
  EgammaHLTEcalIsolation* test_;
};

