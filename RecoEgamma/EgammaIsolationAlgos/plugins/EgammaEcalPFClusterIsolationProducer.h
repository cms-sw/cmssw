#ifndef EgammaIsolationProducers_EgammaEcalPFClusterIsolationProducer_h
#define EgammaIsolationProducers_EgammaEcalPFClusterIsolationProducer_h

//*****************************************************************************
// File:      EgammaEcalPFClusterIsolationProducer.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matteo Sani
// Institute: UCSD
//*****************************************************************************

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

template<typename T1>
class EgammaEcalPFClusterIsolationProducer : public edm::stream::EDProducer<> {
 public:

  typedef std::vector<T1> T1Collection;
  typedef edm::Ref<T1Collection> T1Ref;
  explicit EgammaEcalPFClusterIsolationProducer(const edm::ParameterSet&);
  ~EgammaEcalPFClusterIsolationProducer();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
 
  virtual void produce(edm::Event&, const edm::EventSetup&);
 private:

  const edm::EDGetTokenT<T1Collection> emObjectProducer_;
  const edm::EDGetTokenT<reco::PFClusterCollection> pfClusterProducer_;
  
  const double drMax_;
  const double drVetoBarrel_;
  const double drVetoEndcap_;
  const double etaStripBarrel_;
  const double etaStripEndcap_;
  const double energyBarrel_;
  const double energyEndcap_;
};

#endif
