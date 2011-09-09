#ifndef ECAL_LASER_HANDLER_H
#define ECAL_LASER_HANDLER_H

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



#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"

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

namespace popcon
{
  class EcalLaserHandler : public popcon::PopConSourceHandler<EcalLaserAPDPNRatios>
  {
    
  public:
    void getNewObjects();
    double diff(float x, float old_x);
    ~EcalLaserHandler(); 
    EcalLaserHandler(edm::ParameterSet const & ); 
    
    EcalCondDBInterface* econn;
    std::string id() const { return m_name;}
    void notifyProblems(const EcalLaserAPDPNRatios::EcalLaserAPDPNpair &old,
			const EcalLaserAPDPNRatios::EcalLaserAPDPNpair &current,
			int hashedIndex, const std::string &reason);
    bool checkAPDPN(const EcalLaserAPDPNRatios::EcalLaserAPDPNpair &old,
		    const EcalLaserAPDPNRatios::EcalLaserAPDPNpair &current,
		    int hashedIndex);
    bool checkAPDPNs(const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap &laserMap,
		     const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap &apdpns_popcon);

    void dumpBarrelPayload(EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap const &laserMap);
    void dumpEndcapPayload(EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap const &laserMap);
    
  private:
    const EcalLaserAPDPNRatios * myapdpns;
    unsigned long m_sequences;
    std::string m_sid;
    std::string m_user;
    std::string m_pass;
    std::string m_name;
    std::string m_maxtime; 
    bool        m_debug;
    
  };
}
#endif
