// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTClusterShapeProducer
// 
/**\class EgammaHLTClusterShapeProducer EgammaHLTClusterShapeProducer.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTClusterShapeProducer.h
*/
//
// Original Author:  Roberto Covarelli (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTClusterShapeProducer.h,v 1.1 2009/01/15 14:28:27 covarell Exp $
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTClusterShapeProducer : public edm::EDProducer {
public:
  explicit EgammaHLTClusterShapeProducer(const edm::ParameterSet&);
  ~EgammaHLTClusterShapeProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  virtual void produce(edm::Event&, const edm::EventSetup&);
private:
  // ----------member data ---------------------------
  
  edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  edm::EDGetTokenT<EcalRecHitCollection>  ecalRechitEBToken_;
  edm::EDGetTokenT<EcalRecHitCollection>  ecalRechitEEToken_;
  bool EtaOrIeta_;

  edm::ParameterSet conf_;
};

