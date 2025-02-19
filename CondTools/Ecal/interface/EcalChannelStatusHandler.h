#ifndef ECAL_CHANNELSTATUS_HANDLER_H
#define ECAL_CHANNELSTATUS_HANDLER_H

#include <typeinfo>
#include <vector>
#include <string>
#include <map>
#include <sstream>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include "TTree.h"
#include "TFile.h"

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

#include "CondTools/Ecal/interface/EcalErrorMask.h"
#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "OnlineDB/EcalCondDB/interface/MonLaserBlueDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNBlueDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingLaserBlueCrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTTErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/all_od_types.h"
#include "OnlineDB/EcalCondDB/interface/all_fe_config_types.h"
#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"

#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"


#include "TProfile2D.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace popcon {
  
  
  class EcalChannelStatusHandler : public popcon::PopConSourceHandler<EcalChannelStatus>
    {
      
    public:
      EcalChannelStatusHandler(edm::ParameterSet const &);
      ~EcalChannelStatusHandler(); 
      
      void getNewObjects();
      void setElectronicsMap(const EcalElectronicsMapping*);
      
      std::string id() const { return m_name;}
      EcalCondDBInterface* econn;

      
      // checks on pedestals
      float checkPedestalValueGain12( EcalPedestals::Item* item);
      float checkPedestalValueGain6( EcalPedestals::Item* item);
      float checkPedestalValueGain1( EcalPedestals::Item* item);
      float checkPedestalRMSGain12( EcalPedestals::Item* item );
      float checkPedestalRMSGain6( EcalPedestals::Item* item );
      float checkPedestalRMSGain1( EcalPedestals::Item* item );
      
      // check which laser sectors are on
      void nBadLaserModules( std::map<EcalLogicID, MonLaserBlueDat> dataset_mon );

      // to mask channels reading from pedestal
      void pedOnlineMasking();
      void pedMasking();
      void laserMasking();
      void physicsMasking();

      // to read the daq configuration
      void daqOut(RunIOV myRun);

      // real analysis
      void pedAnalysis( std::map<EcalLogicID, MonPedestalsDat> dataset_mon, std::map<EcalLogicID, MonCrystalConsistencyDat> wrongGain_mon );
      void laserAnalysis( std::map<EcalLogicID, MonLaserBlueDat> dataset_mon );
      void cosmicsAnalysis( std::map<EcalLogicID, MonPedestalsOnlineDat> pedestalO_mon, std::map<EcalLogicID, MonCrystalConsistencyDat> wrongGain_mon, std::map<EcalLogicID, MonLaserBlueDat> laser_mon, std::map<EcalLogicID, MonOccupancyDat> occupancy_mon );
      
    private:
      
      unsigned int m_firstRun ;
      unsigned int m_lastRun ;
      
      std::string m_location;
      std::string m_gentag;
      std::string m_runtype;			
      std::string m_sid;
      std::string m_user;
      std::string m_pass;
      std::string m_locationsource;
      std::string m_name;

      bool isGoodLaserEBSm[36][2];
      bool isGoodLaserEESm[18][2];
      bool isEBRef1[36][2];
      bool isEBRef2[36][2];
      bool isEERef1[18][2];
      bool isEERef2[18][2];

      EcalElectronicsMapping ecalElectronicsMap_;

      ofstream *ResFileEB;
      ofstream *ResFileEE;
      ofstream *ResFileNewEB;
      ofstream *ResFileNewEE;
      ofstream *daqFile;
      ofstream *daqFile2;

      std::map<DetId, float> maskedOnlinePedEB, maskedOnlinePedEE;
      std::map<DetId, float> maskedPedEB, maskedPedEE;
      std::map<DetId, float> maskedLaserEB, maskedLaserEE;
      std::map<DetId, float> maskedPhysicsEB, maskedPhysicsEE;

      TProfile2D *newBadEB_;
      TProfile2D *newBadEEP_;
      TProfile2D *newBadEEM_;
    };
}
#endif

