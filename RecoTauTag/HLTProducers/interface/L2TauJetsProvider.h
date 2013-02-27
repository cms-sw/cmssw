
#ifndef L2TAUJETSPROVIDER_H
#define L2TAUJETSPROVIDER_H

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
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
//#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"


#include <map>
#include <vector>

class L2TauJetsProvider: public edm::EDProducer {
 public:
  explicit L2TauJetsProvider(const edm::ParameterSet&);
  ~L2TauJetsProvider();
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:
  std::vector<l1extra::L1JetParticleRef> tauCandRefVec;
std::vector<l1extra::L1JetParticleRef> jetCandRefVec;
  std::vector<l1extra::L1JetParticleRef> objL1CandRefVec;
  l1extra::L1JetParticleRef tauCandRef;
    
  typedef std::vector<edm::InputTag> vtag;
  vtag jetSrc;
  edm::InputTag l1ParticlesTau;
  edm::InputTag l1ParticlesJet;
  edm::InputTag tauTrigger;
  double mEt_Min;
  std::map<int, const reco::CaloJet> myL2L1JetsMap; //first is # L1Tau , second is L2 jets
};
#endif
