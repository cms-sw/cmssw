
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTTErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunPNErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunMemChErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunMemTTErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunIOV.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "CondTools/Ecal/interface/EcalErrorMask.h"

#include <iostream>

int  EcalErrorMask::runNb_ = -1;
std::map<EcalLogicID, RunCrystalErrorsDat> EcalErrorMask::mapCrystalErrors_;
std::map<EcalLogicID, RunTTErrorsDat>      EcalErrorMask::mapTTErrors_;
std::map<EcalLogicID, RunPNErrorsDat>      EcalErrorMask::mapPNErrors_;
std::map<EcalLogicID, RunMemChErrorsDat>   EcalErrorMask::mapMemChErrors_;
std::map<EcalLogicID, RunMemTTErrorsDat>   EcalErrorMask::mapMemTTErrors_;

void EcalErrorMask::readDB( EcalCondDBInterface* eConn, RunIOV* runIOV ) throw( std::runtime_error ) {

  if( eConn ) {

    RunIOV validIOV;
    RunTag runTag = runIOV->getRunTag();

    std::string location = runTag.getLocationDef().getLocation();

    std::cout << std::endl;
    std::cout << " RunCrystalErrorsDat: ";
    try {
      eConn->fetchValidDataSet( &EcalErrorMask::mapCrystalErrors_, &validIOV, location, runIOV->getRunNumber() );
      std::cout << "found" << std::endl;
    } catch ( std::runtime_error &e ) {
      std::cout << "not found" << std::endl;
      throw( std::runtime_error( e.what() ) );
    }

    // use the IOV for CrystalErrors as reference
    EcalErrorMask::runNb_ = validIOV.getRunNumber();

    std::cout << " RunTTErrorsDat:      ";
    try {
      eConn->fetchValidDataSet( &EcalErrorMask::mapTTErrors_,      &validIOV, location, runIOV->getRunNumber() );
      std::cout << "found" << std::endl;
    } catch ( std::runtime_error &e ) {
      std::cout << "not found" << std::endl;
    }
    std::cout << " RunPNErrorsDat:      ";
    try {
      eConn->fetchValidDataSet( &EcalErrorMask::mapPNErrors_,      &validIOV, location, runIOV->getRunNumber() );
      std::cout << "found" << std::endl;
    } catch ( std::runtime_error &e ) {
      std::cout << "not found" << std::endl;
    }
    std::cout << " RunMemChErrorsDat:   ";
    try {
      eConn->fetchValidDataSet( &EcalErrorMask::mapMemChErrors_,   &validIOV, location, runIOV->getRunNumber() );
      std::cout << "found" << std::endl;
    } catch ( std::runtime_error &e ) {
      std::cout << "not found" << std::endl;
    }
    std::cout << " RunMemTTErrorsDat:   ";
    try {
      eConn->fetchValidDataSet( &EcalErrorMask::mapMemTTErrors_,   &validIOV, location, runIOV->getRunNumber() );
      std::cout << "found" << std::endl;
    } catch ( std::runtime_error &e ) {
      std::cout << "not found" << std::endl;
    }

    std::cout << std::endl;

  }

}

void EcalErrorMask::fetchDataSet( std::map< EcalLogicID, RunCrystalErrorsDat>* fillMap ) {

  fillMap->clear();
  *fillMap = EcalErrorMask::mapCrystalErrors_;
  return;

}

void EcalErrorMask::fetchDataSet( std::map< EcalLogicID, RunTTErrorsDat>* fillMap ) {

  fillMap->clear();
  *fillMap = EcalErrorMask::mapTTErrors_;
  return;

}

void EcalErrorMask::fetchDataSet( std::map< EcalLogicID, RunPNErrorsDat>* fillMap ) {

  fillMap->clear();
  *fillMap = EcalErrorMask::mapPNErrors_;
  return;

}

void EcalErrorMask::fetchDataSet( std::map< EcalLogicID, RunMemChErrorsDat>* fillMap ) {

  fillMap->clear();
  *fillMap = EcalErrorMask::mapMemChErrors_;
  return;

}

void EcalErrorMask::fetchDataSet( std::map< EcalLogicID, RunMemTTErrorsDat>* fillMap ) {

  fillMap->clear();
  *fillMap = EcalErrorMask::mapMemTTErrors_;
  return;

}

