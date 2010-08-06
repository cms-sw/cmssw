/*
 * \file EcalDQMStatusWriter.cc
 *
 * $Date: 2010/08/06 15:34:49 $
 * $Revision: 1.1 $
 * \author G. Della Ricca
 *
*/

#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <time.h>
#include <unistd.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "DQM/EcalCommon/interface/EcalDQMStatusDictionary.h"

#include "DQM/EcalCommon/interface/EcalDQMStatusWriter.h"

EcalDQMStatusWriter::EcalDQMStatusWriter(const edm::ParameterSet& c) {

  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters toPut=c.getParameter<Parameters>("toPut");

  for (Parameters::iterator itToPut=toPut.begin(); itToPut!=toPut.end(); itToPut++) {
      inpFileName_.push_back(itToPut->getUntrackedParameter<std::string>("inputFile"));
      objectName_.push_back(itToPut->getUntrackedParameter<std::string>("conditionType"));
      since_.push_back(itToPut->getUntrackedParameter<unsigned int>("since"));
  }

}

void EcalDQMStatusWriter::endJob() {

  edm::Service<cond::service::PoolDBOutputService> dbservice;
  if ( !dbservice.isAvailable() ){
    std::cout << "PoolDBOutputService is unavailable" << std::endl;
    return;
  }

  bool toAppend=false;

  for (unsigned int i=0; i<objectName_.size(); i++) {

      cond::Time_t newTime;

      if ( dbservice->isNewTagRequest( objectName_[i]+std::string("Rcd") ) ) {
        // This is the first object for this tag.
        // Append mode should be off.
        // newTime is the end of this new objects IOV.
        newTime = dbservice->beginOfTime();
      } else {
        // There should already be an object in the DB for this tag.
        // Append IOV mode should be on.
        // newTime is the beginning of this new objects IOV.
        toAppend=true;
        newTime = (cond::Time_t) since_[i];
      }

      std::cout << "Reading " << objectName_[i] << " from file and writing to DB with newTime " << newTime << std::endl;

      if (objectName_[i]  ==  "EcalDQMChannelStatus") {

        EcalDQMChannelStatus* status = readEcalDQMChannelStatusFromFile(inpFileName_[i].c_str());

        if ( !toAppend ) {
          dbservice->createNewIOV<EcalDQMChannelStatus>(status, newTime, dbservice->endOfTime(), "EcalDQMChannelStatusRcd");
        } else {
          dbservice->appendSinceTime<EcalDQMChannelStatus>(status, newTime, "EcalDQMChannelStatusRcd");
        }

      } else if (objectName_[i]  ==  "EcalDQMTowerStatus") {

        EcalDQMTowerStatus* status = readEcalDQMTowerStatusFromFile(inpFileName_[i].c_str());

        if ( !toAppend ) {
          dbservice->createNewIOV<EcalDQMTowerStatus>(status, newTime, dbservice->endOfTime(), "EcalDQMTowerStatusRcd");
        } else {
          dbservice->appendSinceTime<EcalDQMTowerStatus>(status, newTime, "EcalDQMTowerStatusRcd");
        }

      } else {

        std::cout << "Object " << objectName_[i]  << " is not supported by this program." << std::endl;

      }

  }

}

void EcalDQMStatusWriter::analyze(const edm::Event& e, const edm::EventSetup& c) {

}

EcalDQMStatusWriter::~EcalDQMStatusWriter() {

}

EcalDQMChannelStatus* EcalDQMStatusWriter::readEcalDQMChannelStatusFromFile(const char* inputFile) {

  EcalDQMChannelStatus* status = new EcalDQMChannelStatus();

  // barrel
  for (int ie=-EBDetId::MAX_IETA; ie<=EBDetId::MAX_IETA; ie++) {
  if ( ie==0 ) continue;
    for (int ip=EBDetId::MIN_IPHI; ip<=EBDetId::MAX_IPHI; ip++) {
      if ( EBDetId::validDetId(ie, ip) ) {
        EBDetId id(ie, ip);
        status->setValue(id, 0);
      }
    }
  }

  // endcap
  for (int ix=EEDetId::IX_MIN; ix<=EEDetId::IX_MAX; ix++) {
    for (int iy=EEDetId::IY_MIN; iy<=EEDetId::IY_MAX; iy++) {
      if ( EEDetId::validDetId(ix, iy, +1) ) {
        EEDetId id(ix, iy, +1);
        status->setValue(id, 0);
      }
      if ( EEDetId::validDetId(ix, iy, -1) ) {
        EEDetId id(ix, iy, -1);
        status->setValue(id, 0);
      }
    }
  }

  std::cout << "Reading channel status from file " << inputFile << std::endl;
  FILE *ifile = fopen( inputFile ,"r" );

  if ( !ifile ) throw cms::Exception ("Cannot open file") ;

  char line[256];

  while ( fgets(line, 255, ifile) ) {

    std::string EBorEE = "";
    int hashedIndex = 0;
    int chStatus = 0;

    std::stringstream aStrStream;

    aStrStream << line;
    aStrStream >> EBorEE;

    if ( EBorEE == "EB" ) {

      aStrStream >> hashedIndex >> chStatus;
      chStatus = convert(chStatus);
      EBDetId aEBDetId=EBDetId::unhashIndex(hashedIndex);

      if ( chStatus!=0 ) std::cout << EBorEE << " hashedIndex " << hashedIndex << " status " <<  chStatus << std::endl;
      status->setValue(aEBDetId, chStatus);

    } else if ( EBorEE == "EE" ) {

      int ix, iy, iz;

      aStrStream >> ix >> iy >> iz >> chStatus;
      chStatus = convert(chStatus);
      EEDetId aEEDetId(ix, iy, iz);

      hashedIndex = aEEDetId.hashedIndex();
      if ( chStatus!=0 ) std::cout << EBorEE << " hashedIndex " << hashedIndex << " status " <<  chStatus << std::endl;
      status->setValue(aEEDetId, chStatus);

    }

  }

  fclose(ifile);

  return status;

}

EcalDQMTowerStatus* EcalDQMStatusWriter::readEcalDQMTowerStatusFromFile(const char* inputFile) {

  EcalDQMTowerStatus* status = new EcalDQMTowerStatus();

  // barrel
  for (int i=EcalTrigTowerDetId::MIN_I; i<=EcalTrigTowerDetId::MAX_I; i++) {
    for (int j=EcalTrigTowerDetId::MIN_J; j<=EcalTrigTowerDetId::MAX_J; j++) {
      if ( EcalTrigTowerDetId::validDetId(0, EcalBarrel, i, j) ) {
        EcalTrigTowerDetId id(0, EcalBarrel, i, j);
        status->setValue(id, 0);
      }
    }
  }
  
  // endcap
  for (int ix=EcalScDetId::IX_MIN; ix<=EcalScDetId::IX_MAX; ix++) {
    for (int iy=EcalScDetId::IY_MIN; iy<=EcalScDetId::IY_MAX; iy++) {
      if ( EcalScDetId::validDetId(ix, iy, +1) ) {
        EcalScDetId id(ix, iy, +1);
        status->setValue(id, 0);
      }
      if ( EcalScDetId::validDetId(ix, iy, -1) ) {
        EcalScDetId id(ix, iy, -1);
        status->setValue(id, 0);
      }
    }
  }

  std::cout << "Reading channel status from file " << inputFile << std::endl;
  FILE *ifile = fopen( inputFile ,"r" );

  if ( !ifile ) throw cms::Exception ("Cannot open file") ;

  char line[256];

  while ( fgets(line, 255, ifile) ) {

  }

  fclose(ifile);

  return status;

}

int EcalDQMStatusWriter::convert(int chStatus) {

  if ( chStatus == 1 || (chStatus >= 8 && chStatus <= 12 )) {

    chStatus = 0;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LASER_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LASER_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LASER_TIMING_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LASER_TIMING_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LED_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LED_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LED_TIMING_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LED_TIMING_RMS_ERROR;

  } else if ( chStatus == 2 ) {

    chStatus = 0;
    chStatus |= 1 << EcalDQMStatusHelper::LASER_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LASER_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LASER_TIMING_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LASER_TIMING_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LED_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LED_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LED_TIMING_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LED_TIMING_RMS_ERROR;

  } else if ( chStatus == 3 ) {

    chStatus = 0;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING;

  } else if ( chStatus == 4 ) {

    chStatus = 0;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR;

  } else if ( chStatus == 13 || chStatus == 14 ) {

    chStatus = 0;
    chStatus |= 1 << EcalDQMStatusHelper::STATUS_FLAG_ERROR;

  } else if ( (chStatus&0x3f) == 13 || (chStatus&0x7f) == 13 ||
              (chStatus&0x3f) == 14 || (chStatus&0x7f) == 14 ) {

    chStatus = 0;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LASER_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LASER_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LASER_TIMING_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LASER_TIMING_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LED_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LED_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LED_TIMING_MEAN_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::LED_TIMING_RMS_ERROR;
    chStatus |= 1 << EcalDQMStatusHelper::STATUS_FLAG_ERROR;

  }

  return( chStatus );

}

