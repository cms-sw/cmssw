#ifndef RecoEcal_EgammaClusterProducers_EgammaHLTHybridClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_EgammaHLTHybridClusterProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEcal/EgammaClusterAlgos/interface/HybridClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "DataFormats/Math/interface/RectangularEtaPhiRegion.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTHybridClusterProducer : public edm::stream::EDProducer<> {
public:
  EgammaHLTHybridClusterProducer(const edm::ParameterSet& ps);
  ~EgammaHLTHybridClusterProducer() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const std::string basicclusterCollection_;
  const std::string superclusterCollection_;
  const edm::EDGetTokenT<EcalRecHitCollection> hittoken_;
  const edm::InputTag hitcollection_;

  const edm::EDGetTokenT<l1extra::L1EmParticleCollection> l1TagIsolated_;
  const edm::EDGetTokenT<l1extra::L1EmParticleCollection> l1TagNonIsolated_;

  const bool doIsolated_;
  const double l1LowerThr_;
  const double l1UpperThr_;
  const double l1LowerThrIgnoreIsolation_;

  const double regionEtaMargin_;
  const double regionPhiMargin_;

  const PositionCalc posCalculator_;  // position calculation algorithm
  HybridClusterAlgo* const hybrid_p;  // clustering algorithm
};
#endif
