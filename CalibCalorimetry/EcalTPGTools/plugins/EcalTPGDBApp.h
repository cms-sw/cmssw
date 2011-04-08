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

//#include "OnlineDB/EcalCondDB/interface/FEConfigSpikeDat.h"

using namespace std;

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
  int  writeToConfDB_TPGLinearCoef(const  map<EcalLogicID, FEConfigLinDat> & linset, 
				   const  map<EcalLogicID, FEConfigLinParamDat> & linparamset, int iovId, string tag) ; 
  int  writeToConfDB_TPGLUT(const  map<EcalLogicID, FEConfigLUTGroupDat> & lutgroup, const  map<EcalLogicID, FEConfigLUTDat> & lutdat, 
			     const  map<EcalLogicID, FEConfigLUTParamDat> & lutparamset, int iovId, string tag) ;
  int  writeToConfDB_TPGWeight(const  map<EcalLogicID, FEConfigWeightGroupDat> & lutgroup, const  map<EcalLogicID, FEConfigWeightDat> & lutdat, 
			    int iovId, string tag) ;
  int  writeToConfDB_TPGFgr(const  map<EcalLogicID, FEConfigFgrGroupDat> & lutgroup, const  map<EcalLogicID, FEConfigFgrDat> & lutdat, 
			    const  map<EcalLogicID, FEConfigFgrParamDat> & fgrparamset,
			    const  map<EcalLogicID, FEConfigFgrEETowerDat> & dataset3,  
			    const  map<EcalLogicID, FEConfigFgrEEStripDat> & dataset4,
			    int iovId, string tag) ;
  int  writeToConfDB_Spike(const  map<EcalLogicID, FEConfigSpikeDat> & spikegroupset, string tag);
  int  writeToConfDB_Delay(const  map<EcalLogicID, FEConfigTimingDat> & delaygroupset, string tag); // modif here 31/1/2011

  int writeToConfDB_TPGSliding(const  map<EcalLogicID, FEConfigSlidingDat> & sliset, int iovId, string tag) ;
  
  void readFromConfDB_TPGPedestals(int iconf_req) ;
  int readFromCondDB_Pedestals(map<EcalLogicID, MonPedestalsDat> & pedset, int runNb) ;
  int writeToConfDB_TPGMain(int ped, int lin, int lut, int fgr, int sli, int wei, int spi, int tim, int bxt, int btt, int bst, string tag, int ver) ;

  
 private:
  
  uint64_t startmicros;
  uint64_t endmicros;
  run_t startrun;
  run_t endrun;
  
  void printTag( const RunTag* tag) const ;
  void printIOV( const RunIOV* iov) const ;
  
};

#endif

