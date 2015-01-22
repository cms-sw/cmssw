// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTPhotonTrackIsolationProducersRegional
// 
/**\class EgammaHLTPhotonTrackIsolationProducersRegional EgammaHLTPhotonTrackIsolationProducersRegional.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTPhotonTrackIsolationProducersRegional.h
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id: EgammaHLTPhotonTrackIsolationProducersRegional.h,v 1.1 2007/03/23 17:22:54 ghezzi Exp $
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

#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTTrackIsolation.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTPhotonTrackIsolationProducersRegional : public edm::global::EDProducer<> {
   public:
      explicit EgammaHLTPhotonTrackIsolationProducersRegional(const edm::ParameterSet&);
      ~EgammaHLTPhotonTrackIsolationProducersRegional();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  virtual void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;
  
private:
      // ----------member data ---------------------------

  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  const edm::EDGetTokenT<reco::TrackCollection> trackProducer_;

  //edm::ParameterSet conf_;

  const bool countTracks_;

  const double egTrkIsoPtMin_; 
  const double egTrkIsoConeSize_;
  const double egTrkIsoZSpan_;   
  const double egTrkIsoRSpan_;  
  const double egTrkIsoVetoConeSize_;
  const double egTrkIsoStripBarrel_;
  const double egTrkIsoStripEndcap_;

  EgammaHLTTrackIsolation* test_;
};

