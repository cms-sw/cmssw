#ifndef EgammaHLTProducers_EgammaHLTPFPhotonIsolationProducer_h
#define EgammaHLTProducers_EgammaHLTPFPhotonIsolationProducer_h

//*****************************************************************************
// OrigAuth:  Matteo Sani
// Institute: UCSD
//*****************************************************************************

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class ConfigurationDescriptions;
}


class EgammaHLTPFPhotonIsolationProducer : public edm::EDProducer {
   public:
      explicit EgammaHLTPFPhotonIsolationProducer(const edm::ParameterSet&);
      ~EgammaHLTPFPhotonIsolationProducer();
      
      virtual void produce(edm::Event&, const edm::EventSetup&);
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  edm::InputTag recoEcalCandidateProducer_;
  edm::InputTag pfCandidates_;

  double drMax_;
  double drVetoBarrel_;
  double drVetoEndcap_;
  double etaStripBarrel_;
  double etaStripEndcap_;
  double energyBarrel_;
  double energyEndcap_;
  int pfToUse_;

  bool doRhoCorrection_;
  float rhoScale_;
  float rhoMax_;

  edm::ParameterSet conf_;
};

#endif
