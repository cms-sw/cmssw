#ifndef ECAL_TPGFINEGRAINSTRIP_H
#define ECAL_TPGFINEGRAINSTRIP_H

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

#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainStripEE.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainStripEERcd.h"

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
  class EcalTPGFineGrainStripfromFile : public popcon::PopConSourceHandler<EcalTPGFineGrainStripEE> {

  public:
    void getNewObjects();
    ~EcalTPGFineGrainStripfromFile();
    EcalTPGFineGrainStripfromFile(edm::ParameterSet const & ); 
    
    std::string id() const { return m_name;}

  private:
    std::string m_name;
  };
}
#endif
