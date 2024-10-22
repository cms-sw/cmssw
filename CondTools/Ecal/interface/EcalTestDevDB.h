#ifndef CondTools_Ecal_EcalTestDevDB_h
#define CondTools_Ecal_EcalTestDevDB_h

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "CondCore/CondDB/interface/Exception.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include <string>
#include <map>
#include <memory>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class EcalTestDevDB : public edm::one::EDAnalyzer<> {
public:
  explicit EcalTestDevDB(const edm::ParameterSet& iConfig);
  ~EcalTestDevDB() override;

  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;

  std::shared_ptr<EcalPedestals> generateEcalPedestals();
  std::shared_ptr<EcalADCToGeVConstant> generateEcalADCToGeVConstant();
  std::shared_ptr<EcalIntercalibConstants> generateEcalIntercalibConstants();
  std::shared_ptr<EcalLinearCorrections> generateEcalLinearCorrections();
  std::shared_ptr<EcalGainRatios> generateEcalGainRatios();
  std::shared_ptr<EcalWeightXtalGroups> generateEcalWeightXtalGroups();
  std::shared_ptr<EcalTBWeights> generateEcalTBWeights();
  std::shared_ptr<EcalLaserAPDPNRatios> generateEcalLaserAPDPNRatios(uint32_t i_run);
  std::shared_ptr<EcalLaserAlphas> generateEcalLaserAlphas();
  std::shared_ptr<EcalLaserAPDPNRatiosRef> generateEcalLaserAPDPNRatiosRef();

private:
  std::string m_timetype;
  std::map<std::string, unsigned long long> m_cacheIDs;
  std::map<std::string, std::string> m_records;
  unsigned long m_firstRun;
  unsigned long m_lastRun;
  unsigned int m_interval;
};

#endif
