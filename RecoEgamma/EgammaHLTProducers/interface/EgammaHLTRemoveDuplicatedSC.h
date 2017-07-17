#ifndef RecoEgamma_EgammaHLTProducers_EgammaHLTRemoveDuplicatedSC_h
#define RecoEgamma_EgammaHLTProducers_EgammaHLTRemoveDuplicatedSC_h

// -*- C++ -*-
//
// Package:    EgammaHLTRemoveDuplicatedSC
// Class:      EgammaHLTRemoveDuplicatedSC
// 
// Description: Remove from the L1NonIso SCs those SCs that are already
// there in the L1Iso SCs.
//
// Original Author:  Alessio Ghezzi
//         Created:  Fri Jan 9 11:50 CEST 2009
//

#include <memory>
#include <string>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTRemoveDuplicatedSC : public edm::EDProducer {
  
 public:
  explicit EgammaHLTRemoveDuplicatedSC(const edm::ParameterSet&);
  ~EgammaHLTRemoveDuplicatedSC();
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  // vars to get products
  edm::EDGetTokenT<reco::SuperClusterCollection> sCInputProducer_;
  edm::EDGetTokenT<reco::SuperClusterCollection> alreadyExistingSC_;
  
  std::string outputCollection_;
  
};
#endif
