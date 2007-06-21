#ifndef ECALPEDESTALTRANSFER_H
#define ECALPEDESTALTRANSFER_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include <string>
#include <map>
#include <iostream>
#include <vector>
#include <time.h>

using namespace std;
using namespace oracle::occi;

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class  EcalPedestalTransfer : public edm::EDAnalyzer {
 public:
  EcalCondDBInterface* econn;

  explicit  EcalPedestalTransfer(const edm::ParameterSet& iConfig );
  ~EcalPedestalTransfer();


  virtual void analyze( const edm::Event& evt, const edm::EventSetup& evtSetup);


  

 private:
 
  //  EcalWeightXtalGroups* generateEcalWeightXtalGroups();
  // EcalTBWeights* generateEcalTBWeights();
  // EcalADCToGeVConstant* generateEcalADCToGeVConstant();
  // EcalIntercalibConstants* generateEcalIntercalibConstants();
  // EcalGainRatios* generateEcalGainRatios();
  std::string m_timetype;
  std::map<std::string, unsigned long long> m_cacheIDs;
  std::map<std::string, std::string> m_records;
  unsigned long m_firstRun ;
  unsigned long m_lastRun ;

  std::string sid;
  std::string user;
  std::string pass;


};

#endif
