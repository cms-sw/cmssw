#ifndef RecoEcal_EgammaClusterProducers_PreshowerClusterShapeProducer_h
#define RecoEcal_EgammaClusterProducers_PreshowerClusterShapeProducer_h

#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/EgammaReco/interface/PreshowerClusterShape.h"
#include "RecoEcal/EgammaClusterAlgos/interface/EndcapPiZeroDiscriminatorAlgo.h"

// authors A. Kyriakis, D. Maletic

class PreshowerClusterShapeProducer : public edm::stream::EDProducer<> {
public:
  typedef math::XYZPoint Point;

  explicit PreshowerClusterShapeProducer(const edm::ParameterSet& ps);

  ~PreshowerClusterShapeProducer() override;

  void produce(edm::Event& evt, const edm::EventSetup& es) override;

private:
  int nEvt_;  // internal counter of events

  //clustering parameters:

  edm::EDGetTokenT<EcalRecHitCollection> preshHitToken_;                // name of module/plugin/producer
                                                                        // producing hits
  edm::EDGetTokenT<reco::SuperClusterCollection> endcapSClusterToken_;  // likewise for producer
                                                                        // of endcap superclusters
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;

  std::string PreshowerClusterShapeCollectionX_;
  std::string PreshowerClusterShapeCollectionY_;

  EndcapPiZeroDiscriminatorAlgo* presh_pi0_algo;  // algorithm doing the real work
};
#endif
