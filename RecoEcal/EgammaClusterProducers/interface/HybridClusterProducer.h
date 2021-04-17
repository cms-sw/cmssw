#ifndef RecoEcal_EgammaClusterProducers_HybridClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_HybridClusterProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEcal/EgammaClusterAlgos/interface/HybridClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

//

class HybridClusterProducer : public edm::stream::EDProducer<> {
public:
  HybridClusterProducer(const edm::ParameterSet& ps);

  ~HybridClusterProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  int nEvt_;  // internal counter of events

  std::string basicclusterCollection_;
  std::string superclusterCollection_;

  edm::EDGetTokenT<EcalRecHitCollection> hitsToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geoToken_;
  edm::ESGetToken<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd> sevLvToken_;

  HybridClusterAlgo* hybrid_p;  // clustering algorithm
  PositionCalc posCalculator_;  // position calculation algorithm
};

#endif
