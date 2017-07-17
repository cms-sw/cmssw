// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTR9IDProducer
// 
/**\class EgammaHLTR9IDProducer EgammaHLTR9IDProducer.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTR9IDProducer.h
*/
//
// Original Author:  Roberto Covarelli (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTR9Producer.h,v 1.2 2010/06/10 16:19:31 ghezzi Exp $
//         modified by Chris Tully (Princeton)
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

namespace edm {
  class ConfigurationDescriptions;
}

class RecoEcalCandidateProducers;

class EgammaHLTR9IDProducer : public edm::global::EDProducer<> {
public:
  explicit EgammaHLTR9IDProducer(const edm::ParameterSet&);
  ~EgammaHLTR9IDProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------member data ---------------------------
  
  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  const edm::EDGetTokenT<EcalRecHitCollection> ecalRechitEBToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> ecalRechitEEToken_;
};

