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

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTElectronCombinedIsolationProducer : public edm::EDProducer {
public:
  explicit EgammaHLTElectronCombinedIsolationProducer(const edm::ParameterSet&);
  ~EgammaHLTElectronCombinedIsolationProducer();
  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:

  edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateProducer_;
  edm::EDGetTokenT<reco::ElectronCollection> electronProducer_;
  std::vector<edm::EDGetTokenT<reco::RecoEcalCandidateIsolationMap> > CaloIsolTag_;
  edm::EDGetTokenT<reco::ElectronIsolationMap> TrackIsolTag_;
  
  std::vector<double> CaloIsolWeight_;
  double TrackIsolWeight_;
  edm::ParameterSet conf_;
};

