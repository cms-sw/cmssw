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
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTTrackIsolation.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

class EgammaHLTPhotonTrackIsolationProducersRegional : public edm::EDProducer {
   public:
      explicit EgammaHLTPhotonTrackIsolationProducersRegional(const edm::ParameterSet&);
      ~EgammaHLTPhotonTrackIsolationProducersRegional();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  edm::EDGetTokenT<reco::TrackCollection> trackProducer_;

  edm::ParameterSet conf_;

  bool countTracks_;

  double egTrkIsoPtMin_; 
  double egTrkIsoConeSize_;
  double egTrkIsoZSpan_;   
  double egTrkIsoRSpan_;  
  double egTrkIsoVetoConeSize_;

  EgammaHLTTrackIsolation* test_;

};

