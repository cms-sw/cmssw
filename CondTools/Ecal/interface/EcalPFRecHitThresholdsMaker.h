#ifndef ECALPFRECHITTHRESHOLDSMAKER_H
#define ECALPFRECHITTHRESHOLDSMAKER_H

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "CondCore/CondDB/interface/Exception.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include <string>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class EcalPedestalsRcd;
class EcalADCToGeVConstant;
class EcalADCToGeVConstantRcd;
class EcalIntercalibConstantsRcd;
class EcalLaserDbService;
class EcalLaserDbRecord;
class EcalPFRecHitThresholdsMaker : public edm::one::EDAnalyzer<> {
public:
  explicit EcalPFRecHitThresholdsMaker(const edm::ParameterSet& iConfig);
  ~EcalPFRecHitThresholdsMaker() override;

  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;

private:
  std::string m_timetype;
  double m_nsigma;
  edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> ecalPedestalsToken_;
  edm::ESGetToken<EcalADCToGeVConstant, EcalADCToGeVConstantRcd> ecalADCToGeVConstantToken_;
  edm::ESGetToken<EcalIntercalibConstants, EcalIntercalibConstantsRcd> ecalIntercalibConstantsToken_;
  edm::ESGetToken<EcalLaserDbService, EcalLaserDbRecord> ecalLaserDbServiceToken_;
};

#endif
