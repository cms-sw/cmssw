#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "CondTools/Ecal/interface/EcalErrorMask.h"

#include <iostream>

void EcalErrorMask::readDB(EcalCondDBInterface* eConn, RunIOV* runIOV) noexcept(false) {
  if (eConn) {
    RunIOV validIOV;
    RunTag runTag = runIOV->getRunTag();

    std::string location = runTag.getLocationDef().getLocation();

    std::cout << std::endl;
    std::cout << " RunCrystalErrorsDat: ";
    try {
      eConn->fetchValidDataSet(&mapCrystalErrors_, &validIOV, location, runIOV->getRunNumber());
      std::cout << "found" << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << "not found" << std::endl;
      throw(std::runtime_error(e.what()));
    }

    // use the IOV for CrystalErrors as reference
    runNb_ = validIOV.getRunNumber();

    std::cout << " RunTTErrorsDat:      ";
    try {
      eConn->fetchValidDataSet(&mapTTErrors_, &validIOV, location, runIOV->getRunNumber());
      std::cout << "found" << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << "not found" << std::endl;
    }
    std::cout << " RunPNErrorsDat:      ";
    try {
      eConn->fetchValidDataSet(&mapPNErrors_, &validIOV, location, runIOV->getRunNumber());
      std::cout << "found" << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << "not found" << std::endl;
    }
    std::cout << " RunMemChErrorsDat:   ";
    try {
      eConn->fetchValidDataSet(&mapMemChErrors_, &validIOV, location, runIOV->getRunNumber());
      std::cout << "found" << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << "not found" << std::endl;
    }
    std::cout << " RunMemTTErrorsDat:   ";
    try {
      eConn->fetchValidDataSet(&mapMemTTErrors_, &validIOV, location, runIOV->getRunNumber());
      std::cout << "found" << std::endl;
    } catch (std::runtime_error& e) {
      std::cout << "not found" << std::endl;
    }

    std::cout << std::endl;
  }
}

void EcalErrorMask::fetchDataSet(std::map<EcalLogicID, RunCrystalErrorsDat>* fillMap) {
  fillMap->clear();
  *fillMap = mapCrystalErrors_;
  return;
}

void EcalErrorMask::fetchDataSet(std::map<EcalLogicID, RunTTErrorsDat>* fillMap) {
  fillMap->clear();
  *fillMap = mapTTErrors_;
  return;
}

void EcalErrorMask::fetchDataSet(std::map<EcalLogicID, RunPNErrorsDat>* fillMap) {
  fillMap->clear();
  *fillMap = mapPNErrors_;
  return;
}

void EcalErrorMask::fetchDataSet(std::map<EcalLogicID, RunMemChErrorsDat>* fillMap) {
  fillMap->clear();
  *fillMap = mapMemChErrors_;
  return;
}

void EcalErrorMask::fetchDataSet(std::map<EcalLogicID, RunMemTTErrorsDat>* fillMap) {
  fillMap->clear();
  *fillMap = mapMemTTErrors_;
  return;
}
