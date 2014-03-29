#ifndef L1HLTTAUMATCHING_H
#define L1HLTTAUMATCHING_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"


#include <map>
#include <vector>
class L1HLTTauMatching: public edm::EDProducer {
 public:
  explicit L1HLTTauMatching(const edm::ParameterSet&);
  ~L1HLTTauMatching();
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:
  std::vector<l1extra::L1JetParticleRef> tauCandRefVec;
  std::vector<l1extra::L1JetParticleRef> jetCandRefVec;
  std::vector<l1extra::L1JetParticleRef> objL1CandRefVec;
  l1extra::L1JetParticleRef tauCandRef;
    
  edm::EDGetTokenT<reco::PFTauCollection> jetSrc;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tauTrigger;
  double mEt_Min;
};
#endif
