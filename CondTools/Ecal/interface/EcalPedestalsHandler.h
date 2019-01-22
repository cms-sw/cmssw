#ifndef ECAL_PEDESTALS_HANDLER_H
#define ECAL_PEDESTALS_HANDLER_H

#include <vector>
#include <typeinfo>
#include <string>
#include <map>
#include <iostream>
#include <ctime>

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



#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"

#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
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

  class EcalPedestalsHandler : public popcon::PopConSourceHandler<EcalPedestals> {

  public:
    EcalPedestalsHandler(edm::ParameterSet const & );
    ~EcalPedestalsHandler() override; 
    bool checkPedestal(EcalPedestals::Item* item);
    void getNewObjects() override;
    void getNewObjectsP5();
    void getNewObjectsH2();
    void readPedestalFile();
    void readPedestalMC();
    void readPedestalTree();
    void readPedestalTimestamp();
    void readPedestal2017();
    std::string id() const override { return m_name;}
    EcalCondDBInterface* econn;

  private:
    const EcalPedestals * mypedestals;

    unsigned int m_firstRun ;
    unsigned int m_lastRun ;
			
    std::string m_location;
    std::string m_gentag;
    std::string m_runtag;
    std::string m_sid;
    std::string m_user;
    std::string m_pass;
    std::string m_locationsource;
    std::string m_name;
    std::string m_filename;
    int m_runtype;
    bool m_corrected;

  };
}
#endif

