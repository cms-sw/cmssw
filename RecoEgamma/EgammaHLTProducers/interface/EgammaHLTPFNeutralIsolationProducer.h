#ifndef EgammaHLTProducers_EgammaHLTPFNeutralIsolationProducer_h
#define EgammaHLTProducers_EgammaHLTPFNeutralIsolationProducer_h

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

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTPFNeutralIsolationProducer : public edm::EDProducer {
 public:
  explicit EgammaHLTPFNeutralIsolationProducer(const edm::ParameterSet&);
  ~EgammaHLTPFNeutralIsolationProducer() {};    
      
  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
 
 private:

  edm::EDGetTokenT<reco::ElectronCollection> electronProducer_;
  edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfCandidateProducer_;
  edm::EDGetTokenT<double> rhoProducer_;

  bool useSCRefs_;

  double drMax_;
  double drVetoBarrel_;
  double drVetoEndcap_;
  double etaStripBarrel_;
  double etaStripEndcap_;
  double energyBarrel_;
  double energyEndcap_;
  int pfToUse_;

  float effectiveAreaBarrel_;
  float effectiveAreaEndcap_;
  bool doRhoCorrection_;
  float rhoScale_;
  float rhoMax_;

};

#endif
