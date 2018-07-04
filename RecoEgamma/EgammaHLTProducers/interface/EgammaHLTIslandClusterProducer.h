#ifndef RecoEcal_EgammaClusterProducers_EgammaHLTIslandClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_EgammaHLTIslandClusterProducer_h_

#include <memory>
#include <ctime>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
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

class EgammaHLTIslandClusterProducer : public edm::global::EDProducer<> {
 public:
  EgammaHLTIslandClusterProducer(const edm::ParameterSet& ps);
  ~EgammaHLTIslandClusterProducer() override;
  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:

  IslandClusterAlgo::VerbosityLevel verbosity;

  bool doBarrel_;
  bool doEndcaps_;
  bool doIsolated_;
  
  edm::InputTag barrelHitCollection_;
  edm::InputTag endcapHitCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> barrelHitToken_;
  edm::EDGetTokenT<EcalRecHitCollection> endcapHitToken_;
  
  std::string barrelClusterCollection_;
  std::string endcapClusterCollection_;
  
  edm::EDGetTokenT<l1extra::L1EmParticleCollection> l1TagIsolated_;
  edm::EDGetTokenT<l1extra::L1EmParticleCollection> l1TagNonIsolated_;
  double l1LowerThr_;
  double l1UpperThr_;
  double l1LowerThrIgnoreIsolation_;
  
  double regionEtaMargin_;
  double regionPhiMargin_;
  
  PositionCalc posCalculator_; // position calculation algorithm
  IslandClusterAlgo * island_p;
  
  const EcalRecHitCollection * getCollection(edm::Event& evt,
					     const edm::EDGetTokenT<EcalRecHitCollection>& hitToken) const ;
  
  
  void clusterizeECALPart(edm::Event &evt, const edm::EventSetup &es,
			  const edm::EDGetTokenT<EcalRecHitCollection>& hitToken,
			  const std::string& clusterCollection,
			  const std::vector<RectangularEtaPhiRegion>& regions,
			  const IslandClusterAlgo::EcalPart& ecalPart) const ;
};
#endif
