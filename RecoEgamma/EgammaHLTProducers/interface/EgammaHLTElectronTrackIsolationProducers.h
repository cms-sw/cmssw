// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTElectronTrackIsolationProducers
// 
/**\class EgammaHLTElectronTrackIsolationProducers EgammaHLTElectronTrackIsolationProducers.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTElectronTrackIsolationProducers.h
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
//
// $Id: EgammaHLTElectronTrackIsolationProducers.h,v 1.4 2012/01/23 12:56:37 sharper Exp $
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



//
// class declaration
//

class EgammaHLTElectronTrackIsolationProducers : public edm::EDProducer {
   public:
      explicit EgammaHLTElectronTrackIsolationProducers(const edm::ParameterSet&);
      ~EgammaHLTElectronTrackIsolationProducers();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  edm::InputTag electronProducer_;
  edm::InputTag trackProducer_;
  edm::InputTag recoEcalCandidateProducer_;
  edm::InputTag beamSpotProducer_;

  bool useGsfTrack_;
  bool useSCRefs_;

  double egTrkIsoPtMin_; 
  double egTrkIsoConeSize_;
  double egTrkIsoZSpan_;   
  double egTrkIsoRSpan_;  
  double egTrkIsoVetoConeSizeBarrel_;
  double egTrkIsoVetoConeSizeEndcap_;
  double egTrkIsoStripBarrel_;
  double egTrkIsoStripEndcap_;

  
  
};

