#ifndef CondTools_Ecal_EcalErrorMask_H
#define CondTools_Ecal_EcalErrorMask_H

#include <cstdlib>
#include <map>
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTTErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunPNErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunMemChErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunMemTTErrorsDat.h"

class EcalCondDBInterface;
class EcalLogicID;
class RunIOV;

class EcalErrorMask {
public:
  void readDB(EcalCondDBInterface* eConn, RunIOV* runIOV) noexcept(false);

  void fetchDataSet(std::map<EcalLogicID, RunCrystalErrorsDat>* fillMap);
  void fetchDataSet(std::map<EcalLogicID, RunTTErrorsDat>* fillMap);
  void fetchDataSet(std::map<EcalLogicID, RunPNErrorsDat>* fillMap);
  void fetchDataSet(std::map<EcalLogicID, RunMemChErrorsDat>* fillMap);
  void fetchDataSet(std::map<EcalLogicID, RunMemTTErrorsDat>* fillMap);

private:
  int runNb_;

  std::map<EcalLogicID, RunCrystalErrorsDat> mapCrystalErrors_;
  std::map<EcalLogicID, RunTTErrorsDat> mapTTErrors_;
  std::map<EcalLogicID, RunPNErrorsDat> mapPNErrors_;
  std::map<EcalLogicID, RunMemChErrorsDat> mapMemChErrors_;
  std::map<EcalLogicID, RunMemTTErrorsDat> mapMemTTErrors_;
};

#endif  // CondTools_Ecal_EcalErrorMask_H
