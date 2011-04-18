#ifndef ECAL_SRP_HANDLER_H
#define ECAL_SRP_HANDLER_H

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


#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"
#include "CondFormats/DataRecord/interface/EcalSRSettingsRcd.h"

#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "DataFormats/Provenance/interface/Timestamp.h"


namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace popcon {
  class EcalSRPHandler : public popcon::PopConSourceHandler<EcalSRSettings>
    {
    public:
      EcalSRPHandler(edm::ParameterSet const & );
      ~EcalSRPHandler(); 

      void getNewObjects();
      std::string id() const { return m_name;}
      EcalCondDBInterface* econn;
      void importDccConfigFile(EcalSRSettings& sr, const std::string& filename, bool debug = false);
      void PrintPayload(EcalSRSettings& sr, std::ofstream& fout);
      void ChangePayload(EcalSRSettings& sref, EcalSRSettings& sr );

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
      std::string m_i_tag;
      bool m_debug;
      int m_i_version;
    };
}
#endif

