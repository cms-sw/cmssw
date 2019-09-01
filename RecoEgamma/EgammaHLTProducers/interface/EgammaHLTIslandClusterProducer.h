#ifndef RecoEcal_EgammaClusterProducers_EgammaHLTIslandClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_EgammaHLTIslandClusterProducer_h_

#include <memory>
#include <ctime>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEcal/EgammaClusterAlgos/interface/IslandClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "DataFormats/Math/interface/RectangularEtaPhiRegion.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTIslandClusterProducer : public edm::stream::EDProducer<> {
public:
  EgammaHLTIslandClusterProducer(const edm::ParameterSet& ps);
  ~EgammaHLTIslandClusterProducer() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const bool doBarrel_;
  const bool doEndcaps_;
  const bool doIsolated_;

  const edm::InputTag barrelHitCollection_;
  const edm::InputTag endcapHitCollection_;
  const edm::EDGetTokenT<EcalRecHitCollection> barrelHitToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> endcapHitToken_;

  const std::string barrelClusterCollection_;
  const std::string endcapClusterCollection_;

  const edm::EDGetTokenT<l1extra::L1EmParticleCollection> l1TagIsolated_;
  const edm::EDGetTokenT<l1extra::L1EmParticleCollection> l1TagNonIsolated_;
  const double l1LowerThr_;
  const double l1UpperThr_;
  const double l1LowerThrIgnoreIsolation_;

  const double regionEtaMargin_;
  const double regionPhiMargin_;

  const PositionCalc posCalculator_;  // position calculation algorithm
  const std::string verb_;
  IslandClusterAlgo* const island_p;

  const EcalRecHitCollection* getCollection(edm::Event& evt,
                                            const edm::EDGetTokenT<EcalRecHitCollection>& hitToken) const;

  void clusterizeECALPart(edm::Event& evt,
                          const edm::EventSetup& es,
                          const edm::EDGetTokenT<EcalRecHitCollection>& hitToken,
                          const std::string& clusterCollection,
                          const std::vector<RectangularEtaPhiRegion>& regions,
                          const IslandClusterAlgo::EcalPart& ecalPart) const;
};
#endif
