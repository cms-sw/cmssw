#ifndef EgammaHLTProducers_EgammaHLTEcalPFClusterIsolationProducer_h
#define EgammaHLTProducers_EgammaHLTEcalPFClusterIsolationProducer_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateIsolation.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

template<typename T1>
class HLTEcalPFClusterIsolationProducer : public edm::stream::EDProducer<> {

  typedef std::vector<T1> T1Collection;
  typedef edm::Ref<T1Collection> T1Ref;
  typedef edm::AssociationMap<edm::OneToValue<std::vector<T1>, float > > T1IsolationMap;
  
 public:
  explicit HLTEcalPFClusterIsolationProducer(const edm::ParameterSet&);
  ~HLTEcalPFClusterIsolationProducer();    
      
  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
 private:

  bool computedRVeto(T1Ref candRef, reco::PFClusterRef pfclu);
  
  edm::EDGetTokenT<T1Collection> recoCandidateProducer_;
  const edm::EDGetTokenT<reco::PFClusterCollection> pfClusterProducer_;
  const edm::EDGetTokenT<double> rhoProducer_;
  
  double drVeto2_;
  const double drMax_;
  const double drVetoBarrel_;
  const double drVetoEndcap_;
  const double etaStripBarrel_;
  const double etaStripEndcap_;
  const double energyBarrel_;
  const double energyEndcap_;

  const bool doRhoCorrection_;
  const float rhoMax_;
  const float rhoScale_;
  const float effectiveAreaBarrel_;
  const float effectiveAreaEndcap_;
};

#endif
