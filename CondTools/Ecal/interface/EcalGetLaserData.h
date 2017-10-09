#ifndef ECALGETLASERDATA_H
#define ECALGETLASERDATA_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CondCore/CondDB/interface/Exception.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"

#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include <string>
#include <map>
#include <iostream>
#include <vector>
#include <time.h>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class  EcalGetLaserData : public edm::EDAnalyzer {
 public:
 
  explicit  EcalGetLaserData(const edm::ParameterSet& iConfig );
  ~EcalGetLaserData();
  
  virtual void analyze( const edm::Event& evt, const edm::EventSetup& evtSetup);


 private:
 
  //std::string m_timetype;
  std::map<std::string, unsigned long long> m_cacheIDs;
  std::map<std::string, std::string> m_records;
  //unsigned long m_firstRun ;
  //unsigned long m_lastRun ;

  virtual void beginJob() ;
  virtual void endJob() ;


};

#endif
