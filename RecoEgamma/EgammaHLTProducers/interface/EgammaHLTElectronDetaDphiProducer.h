// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTElectronDetaDphiProducer
// 
/**\class EgammaHLTElectronDetaDphiProducer EgammaHLTElectronDetaDphiProducer.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTElectronDetaDphiProducer.h
*/
//
// Original Author:  Roberto Covarelli (CERN)
//
// $Id: EgammaHLTElectronDetaDphiProducer.h,v 1.1 2009/01/15 14:28:27 covarell Exp $
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

#include "TTree.h"
//
// class declaration
//

class EgammaHLTElectronDetaDphiProducer : public edm::EDProducer {
   public:
      explicit EgammaHLTElectronDetaDphiProducer(const edm::ParameterSet&);
      ~EgammaHLTElectronDetaDphiProducer();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  edm::InputTag electronProducer_;
  edm::ParameterSet conf_;

  bool useTrackProjectionToEcal_;
  edm::InputTag BSProducer_;
  TTree * tree;
  float mydphi;
  Int_t myevent, myrun;
};

