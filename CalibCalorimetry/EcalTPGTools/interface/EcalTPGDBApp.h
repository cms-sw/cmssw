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
  
  EcalTPGDBApp(string host, string sid, string user, string pass, int port) ;
  EcalTPGDBApp(string sid, string user, string pass) ;
  
  inline std::string to_string( char value[])
    {
      std::ostringstream streamOut;
      streamOut << value;
      return streamOut.str();    
    }
  
  int  writeToConfDB_TPGPedestals(const  map<EcalLogicID, FEConfigPedDat> & pedset, int iovId, string tag) ;
  int  writeToConfDB_TPGLinearCoef(const  map<EcalLogicID, FEConfigLinDat> & linset, int iovId, string tag) ; 
  void writeToConfDB_TPGLUT() ;
  void writeToConfDB_TPGWeights(FEConfigWeightGroupDat & weight) ;
  
  void readFromConfDB_TPGPedestals(int iconf_req) ;
  int readFromCondDB_Pedestals(map<EcalLogicID, MonPedestalsDat> & pedset, int runNb) ;
  
  
 private:
  
  uint64_t startmicros;
  uint64_t endmicros;
  run_t startrun;
  run_t endrun;
  
  void printTag( const RunTag* tag) const ;
  void printIOV( const RunIOV* iov) const ;
  
};

#endif

