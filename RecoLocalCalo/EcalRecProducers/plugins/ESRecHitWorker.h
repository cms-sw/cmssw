#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerMaxSample_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerMaxSample_hh

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoLocalCalo/EcalRecProducers/interface/ESRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/ESRecHitSimAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/ESRecHitFitAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/ESRecHitAnalyticAlgo.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "CondFormats/ESObjects/interface/ESChannelStatus.h"
#include "CondFormats/ESObjects/interface/ESGain.h"
#include "CondFormats/ESObjects/interface/ESMIPToGeVConstant.h"
#include "CondFormats/ESObjects/interface/ESTimeSampleWeights.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "CondFormats/ESObjects/interface/ESRecHitRatioCuts.h"
#include "CondFormats/ESObjects/interface/ESAngleCorrectionFactors.h"
#include "CondFormats/DataRecord/interface/ESGainRcd.h"
#include "CondFormats/DataRecord/interface/ESChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/ESMIPToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/ESTimeSampleWeightsRcd.h"
#include "CondFormats/DataRecord/interface/ESPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/ESIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/ESRecHitRatioCutsRcd.h"
#include "CondFormats/DataRecord/interface/ESAngleCorrectionFactorsRcd.h"

#include <vector>

namespace edm {
  class ParameterSet;
  class EventSetup;
}  // namespace edm

class ESRecHitWorker : public ESRecHitWorkerBaseClass {
public:
  ESRecHitWorker(const edm::ParameterSet &ps, edm::ConsumesCollector cc);
  ~ESRecHitWorker() override;

  void set(const edm::EventSetup &es) override;
  bool run(const ESDigiCollection::const_iterator &digi, ESRecHitCollection &result) override;

private:
  int recoAlgo_;
  ESRecHitSimAlgo *algoW_;
  ESRecHitFitAlgo *algoF_;
  ESRecHitAnalyticAlgo *algoA_;

  edm::ESHandle<ESGain> esgain_;
  edm::ESHandle<ESMIPToGeVConstant> esMIPToGeV_;
  edm::ESHandle<ESTimeSampleWeights> esWeights_;
  edm::ESHandle<ESPedestals> esPedestals_;
  edm::ESHandle<ESIntercalibConstants> esMIPs_;
  edm::ESHandle<ESChannelStatus> esChannelStatus_;
  edm::ESHandle<ESRecHitRatioCuts> esRatioCuts_;
  edm::ESHandle<ESAngleCorrectionFactors> esAngleCorrFactors_;
  edm::ESGetToken<ESGain, ESGainRcd> esgainToken_;
  edm::ESGetToken<ESMIPToGeVConstant, ESMIPToGeVConstantRcd> esMIPToGeVToken_;
  edm::ESGetToken<ESTimeSampleWeights, ESTimeSampleWeightsRcd> esWeightsToken_;
  edm::ESGetToken<ESPedestals, ESPedestalsRcd> esPedestalsToken_;
  edm::ESGetToken<ESIntercalibConstants, ESIntercalibConstantsRcd> esMIPsToken_;
  edm::ESGetToken<ESChannelStatus, ESChannelStatusRcd> esChannelStatusToken_;
  edm::ESGetToken<ESRecHitRatioCuts, ESRecHitRatioCutsRcd> esRatioCutsToken_;
  edm::ESGetToken<ESAngleCorrectionFactors, ESAngleCorrectionFactorsRcd> esAngleCorrFactorsToken_;
};
#endif
