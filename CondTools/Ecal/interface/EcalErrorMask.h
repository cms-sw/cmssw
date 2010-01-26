#ifndef EcalErrorMask_H
#define EcalErrorMask_H

#include <cstdlib>
#include <map>

class EcalCondDBInterface;

class EcalLogicID;

class RunCrystalErrorsDat;
class RunTTErrorsDat;
class RunPNErrorsDat;
class RunMemChErrorsDat;
class RunMemTTErrorsDat;
class RunIOV;

class EcalErrorMask {

 public:

  static void readDB( EcalCondDBInterface* eConn, RunIOV* runIOV ) throw( std::runtime_error );

  static void fetchDataSet( std::map< EcalLogicID, RunCrystalErrorsDat>* fillMap );
  static void fetchDataSet( std::map< EcalLogicID, RunTTErrorsDat>* fillMap );
  static void fetchDataSet( std::map< EcalLogicID, RunPNErrorsDat>* fillMap );
  static void fetchDataSet( std::map< EcalLogicID, RunMemChErrorsDat>* fillMap );
  static void fetchDataSet( std::map< EcalLogicID, RunMemTTErrorsDat>* fillMap );

 private:

  static int runNb_;

  static std::map<EcalLogicID, RunCrystalErrorsDat> mapCrystalErrors_;
  static std::map<EcalLogicID, RunTTErrorsDat>      mapTTErrors_;
  static std::map<EcalLogicID, RunPNErrorsDat>      mapPNErrors_;
  static std::map<EcalLogicID, RunMemChErrorsDat>   mapMemChErrors_;
  static std::map<EcalLogicID, RunMemTTErrorsDat>   mapMemTTErrors_;

};

#endif // EcalErrorMask_H
