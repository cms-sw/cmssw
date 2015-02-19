#ifndef EgammaHLTProducers_EgammaHLTPFChargedIsolationProducer_h
#define EgammaHLTProducers_EgammaHLTPFChargedIsolationProducer_h

//
// Original Author:  Matteo Sani (UCSD)
//

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTPFChargedIsolationProducer : public edm::EDProducer {
 public:
  explicit EgammaHLTPFChargedIsolationProducer(const edm::ParameterSet&);
  ~EgammaHLTPFChargedIsolationProducer() {};

  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
 
private:

  edm::EDGetTokenT<reco::ElectronCollection> electronProducer_;
  edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  const edm::EDGetTokenT<reco::PFCandidateCollection> pfCandidateProducer_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotProducer_;
  
  const bool useGsfTrack_;
  const bool useSCRefs_;
  
  const double drMax_;
  const double drVetoBarrel_;
  const double drVetoEndcap_;
  const double ptMin_;
  const double dzMax_;
  const double dxyMax_;
  const int pfToUse_;

};

#endif
