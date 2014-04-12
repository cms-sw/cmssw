#ifndef ECALTESTDEVDB_H
#define ECALTESTDEVDB_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CondCore/DBCommon/interface/Exception.h"

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

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class  EcalTestDevDB : public edm::EDAnalyzer {
 public:
  explicit  EcalTestDevDB(const edm::ParameterSet& iConfig );
  ~EcalTestDevDB();


  virtual void analyze( const edm::Event& evt, const edm::EventSetup& evtSetup);

  EcalPedestals* generateEcalPedestals();
  EcalADCToGeVConstant* generateEcalADCToGeVConstant();
  EcalIntercalibConstants* generateEcalIntercalibConstants();
  EcalLinearCorrections* generateEcalLinearCorrections();
  EcalGainRatios* generateEcalGainRatios();
  EcalWeightXtalGroups* generateEcalWeightXtalGroups();
  EcalTBWeights* generateEcalTBWeights();
  EcalLaserAPDPNRatios* generateEcalLaserAPDPNRatios(uint32_t i_run);
  EcalLaserAlphas* generateEcalLaserAlphas();
  EcalLaserAPDPNRatiosRef* generateEcalLaserAPDPNRatiosRef();
  

 private:
 
  std::string m_timetype;
  std::map<std::string, unsigned long long> m_cacheIDs;
  std::map<std::string, std::string> m_records;
  unsigned long m_firstRun ;
  unsigned long m_lastRun ;
  unsigned int m_interval ;
};

#endif
