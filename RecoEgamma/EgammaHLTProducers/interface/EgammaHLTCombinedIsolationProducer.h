// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTCombinedIsolationProducer
// 
/**\class EgammaHLTCombinedIsolationProducer EgammaHLTCombinedIsolationProducer.cc RecoEgamma/EgammaHLTProducers/interface/EgammaHLTCombinedIsolationProducer.h
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

class EgammaHLTCombinedIsolationProducer : public edm::EDProducer {
   public:
      explicit EgammaHLTCombinedIsolationProducer(const edm::ParameterSet&);
      ~EgammaHLTCombinedIsolationProducer();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

  edm::InputTag recoEcalCandidateProducer_;
  std::vector<edm::InputTag> IsolTag_;
  std::vector<double> IsolWeight_;
  edm::ParameterSet conf_;

};

