#ifndef ECALTPGDBAPP_H
#define ECALTPGDBAPP_H

#include <iostream>
#include <string>
#include <sstream>

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "OnlineDB/EcalCondDB/interface/RunList.h"
#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/EcalCondDB/interface/all_fe_config_types.h"

class EcalTPGDBApp : public EcalCondDBInterface {
 public:
  
  EcalTPGDBApp(std::string host, std::string sid, std::string user, std::string pass, int port) ;
  EcalTPGDBApp(std::string sid, std::string user, std::string pass) ;
  
  inline std::string to_string( char value[])
    {
      std::ostringstream streamOut;
      streamOut << value;
      return streamOut.str();    
    }
  
  int  writeToConfDB_TPGPedestals(const  std::map<EcalLogicID, FEConfigPedDat> & pedset, int iovId, std::string tag) ;
  int  writeToConfDB_TPGLinearCoef(const  std::map<EcalLogicID, FEConfigLinDat> & linset, 
				   const  std::map<EcalLogicID, FEConfigLinParamDat> & linparamset, int iovId, std::string tag) ; 
  int  writeToConfDB_TPGLUT(const  std::map<EcalLogicID, FEConfigLUTGroupDat> & lutgroup, const  std::map<EcalLogicID, FEConfigLUTDat> & lutdat, 
			     const  std::map<EcalLogicID, FEConfigLUTParamDat> & lutparamset, int iovId, std::string tag) ;
  int  writeToConfDB_TPGWeight(const  std::map<EcalLogicID, FEConfigWeightGroupDat> & lutgroup, const  std::map<EcalLogicID, FEConfigWeightDat> & lutdat, 
			    int iovId, std::string tag) ;
  int  writeToConfDB_TPGFgr(const  std::map<EcalLogicID, FEConfigFgrGroupDat> & lutgroup, const  std::map<EcalLogicID, FEConfigFgrDat> & lutdat, 
			    const  std::map<EcalLogicID, FEConfigFgrParamDat> & fgrparamset,
			    const  std::map<EcalLogicID, FEConfigFgrEETowerDat> & dataset3,  
			    const  std::map<EcalLogicID, FEConfigFgrEEStripDat> & dataset4,
			    int iovId, std::string tag) ;
  int writeToConfDB_TPGSliding(const  std::map<EcalLogicID, FEConfigSlidingDat> & sliset, int iovId, std::string tag) ;
  
  void readFromConfDB_TPGPedestals(int iconf_req) ;
  int readFromCondDB_Pedestals(std::map<EcalLogicID, MonPedestalsDat> & pedset, int runNb) ;
  int writeToConfDB_TPGMain(int ped, int lin, int lut, int fgr, int sli, int wei, int bxt, int btt, std::string tag, int ver) ;
  
  
 private:
  
  uint64_t startmicros;
  uint64_t endmicros;
  run_t startrun;
  run_t endrun;
  
  void printTag( const RunTag* tag) const ;
  void printIOV( const RunIOV* iov) const ;
  
};

#endif

