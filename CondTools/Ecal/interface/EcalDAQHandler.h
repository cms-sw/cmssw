#ifndef ECAL_DAQ_HANDLER_H
#define ECAL_DAQ_HANDLER_H

#include <vector>
#include <typeinfo>
#include <string>
#include <map>
#include <iostream>
#include <time.h>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"


#include "CondFormats/EcalObjects/interface/EcalDAQTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDAQTowerStatusRcd.h"

#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/Provenance/interface/Timestamp.h"


namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace popcon {
  class EcalDAQHandler : public popcon::PopConSourceHandler<EcalDAQTowerStatus>
    {
    public:
      EcalDAQHandler(edm::ParameterSet const & );
      ~EcalDAQHandler(); 

      void getNewObjects();
      std::string id() const { return m_name;}
      EcalCondDBInterface* econn;

      int detIDToLogicID(int, int, int);
      uint16_t OffDBStatus( uint16_t dbStatus , int pos ) ;

    private:
      unsigned long m_firstRun ;
      unsigned long m_lastRun ;		
      std::string m_sid;
      std::string m_user;
      std::string m_pass;
      std::string m_name;
      std::string m_location;
      std::string m_runtype;
      std::string m_gentag;
      bool        m_debug;
    };
}
#endif

