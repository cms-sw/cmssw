#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerMaxSample_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerMaxSample_hh

#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoLocalCalo/EcalRecProducers/interface/ESRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/ESRecHitSimAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/ESRecHitFitAlgo.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "CondFormats/ESObjects/interface/ESChannelStatus.h"
#include "CondFormats/ESObjects/interface/ESGain.h"
#include "CondFormats/ESObjects/interface/ESMIPToGeVConstant.h"
#include "CondFormats/ESObjects/interface/ESTBWeights.h"
#include "CondFormats/ESObjects/interface/ESWeightSet.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "CondFormats/ESObjects/interface/ESRecHitRatioCuts.h"

namespace edm {
  class ParameterSet;
  class EventSetup;
  class Event;
}

class ESRecHitWorker : public ESRecHitWorkerBaseClass {

 public:

  ESRecHitWorker(const edm::ParameterSet& ps);
  ~ESRecHitWorker();
  
  void set(const edm::EventSetup& es);
  bool run(const edm::Event& evt, const ESDigiCollection::const_iterator & digi, ESRecHitCollection & result);

 private:

  int recoAlgo_;  
  ESRecHitSimAlgo *algoW_;
  ESRecHitFitAlgo *algoF_;

  edm::ESHandle<ESGain> esgain_;
  edm::ESHandle<ESMIPToGeVConstant> esMIPToGeV_;
  edm::ESHandle<ESTBWeights> esWeights_;
  edm::ESHandle<ESPedestals> esPedestals_;
  edm::ESHandle<ESIntercalibConstants> esMIPs_;
  edm::ESHandle<ESChannelStatus> esChannelStatus_;
  edm::ESHandle<ESRecHitRatioCuts> esRatioCuts_;

};
#endif
