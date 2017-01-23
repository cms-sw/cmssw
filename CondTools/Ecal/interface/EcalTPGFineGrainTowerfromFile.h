#ifndef ECAL_TPGFINEGRAINTOWER_H
#define ECAL_TPGFINEGRAINTOWER_H

#include <vector>
#include <typeinfo>
#include <string>
#include <map>
#include <iostream>
#include <time.h>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainTowerEE.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainTowerEERcd.h"

#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/EcalCondDB/interface/all_lmf_types.h"

#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace popcon {
  class EcalTPGFineGrainTowerfromFile : public popcon::PopConSourceHandler<EcalTPGFineGrainTowerEE> {

  public:
    void getNewObjects();
    ~EcalTPGFineGrainTowerfromFile();
    EcalTPGFineGrainTowerfromFile(edm::ParameterSet const & ); 
    
    std::string id() const { return m_name;}

  private:
    std::string m_name;
    std::string fname;
  };
}
#endif
