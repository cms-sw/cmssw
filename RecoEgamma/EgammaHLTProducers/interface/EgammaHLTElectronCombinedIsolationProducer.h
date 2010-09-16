// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTElectronCombinedIsolationProducer
// 
/**\class EgammaHLTElectronCombinedIsolationProducer EgammaHLTElectronCombinedIsolationProducer.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTElectronCombinedIsolationProducer.h
*/
//



// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class EgammaHLTElectronCombinedIsolationProducer : public edm::EDProducer {
   public:
      explicit EgammaHLTElectronCombinedIsolationProducer(const edm::ParameterSet&);
      ~EgammaHLTElectronCombinedIsolationProducer();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  edm::InputTag recoEcalCandidateProducer_;
  edm::InputTag electronProducer_;
  std::vector<edm::InputTag> CaloIsolTag_;
  std::vector<double> CaloIsolWeight_;
  edm::InputTag TrackIsolTag_;
  double TrackIsolWeight_;
  edm::ParameterSet conf_;

};

