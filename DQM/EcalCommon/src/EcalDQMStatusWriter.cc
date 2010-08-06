/*
 * \file EcalDQMStatusWriter.cc
 *
 * $Date: 2010/08/06 17:05:51 $
 * $Revision: 1.2 $
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

  std::vector<EcalDQMStatusDictionary::codeDef_t> dictionary;
  EcalDQMStatusDictionary::getDictionary( dictionary );

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

  int ii = 0;
  while ( fgets(line, 255, ifile) ) {

    std::string token = "";

    std::stringstream stream;

    ii++;
    stream << line;
    stream >> token;

    if ( strcmp(token.c_str(), "EB") == 0 ) {

      int index, code;
      stream >> index >> code;

      EBDetId id = EBDetId::unhashIndex(index);
      code = convert(code);

      int hashedIndex = id.hashedIndex();
      if ( code !=0 ) std::cout << token << " hashedIndex " << hashedIndex << " status " <<  code << std::endl;
      EcalDQMChannelStatus::const_iterator it = status->find(id);
      if ( it != status->end() ) code |= it->getStatusCode();
      status->setValue(id, code);

    } else if ( strcmp(token.c_str(), "EE") == 0 ) {

      int ix, iy, iz, code;
      stream >> ix >> iy >> iz >> code;

      EEDetId id(ix, iy, iz);
      code = convert(code);

      int hashedIndex = id.hashedIndex();
      if ( code!=0 ) std::cout << token << " hashedIndex " << hashedIndex << " status " <<  code << std::endl;
      EcalDQMChannelStatus::const_iterator it = status->find(id);
      if ( it != status->end() ) code |= it->getStatusCode();
      status->setValue(id, code);

    } else if ( strcmp(token.c_str(), "Crystal") == 0 ) {

      std::string module;
      stream >> module;

      if ( strncmp(module.c_str(), "EB+", 3) == 0 || strncmp(module.c_str(), "EB-", 3) == 0 ) {

#if 0
        int ie, ip;
        std::string word;
        stream >> ie >> ip >> word;

        int ism = atoi( module.substr(2, module.size()-2).c_str() );
        ism = ( ism>=0 ) ? ism : 18-ism;
        int iex = (ism>=1&&ism<=18) ? -ie : +ie;
        int ipx = (ism>=1&&ism<=18) ? ip+20*(ism-1) : 1+(20-ip)+20*(ism-19);
#else
        int index;
        std::string word;
        stream >> index >> word;

        int ism = atoi( module.substr(2, module.size()-2).c_str() );
        ism = ( ism>=0 ) ? ism : 18-ism;
        int ie = (index-1)/20 + 1;
        int ip = (index-1)%20 + 1;
        int iex = (ism>=1&&ism<=18) ? -ie : +ie;
        int ipx = (ism>=1&&ism<=18) ? ip+20*(ism-1) : 1+(20-ip)+20*(ism-19);
#endif

        EBDetId id(iex, ipx);
        int code = -1;
        for (unsigned int i=0; i<dictionary.size(); i++) {
          if ( strcmp(word.c_str(), dictionary[i].desc) == 0 ) {
            code = dictionary[i].code;
          }
        }
        if ( code == -1 ) {
          std::cout << " --> not found in the dictionary: " << word << std::endl;
          continue;
        }

        int hashedIndex = id.hashedIndex();
        if ( code !=0 ) std::cout << module << " hashedIndex " << hashedIndex << " status " <<  code << std::endl;
        EcalDQMChannelStatus::const_iterator it = status->find(id);
        if ( it != status->end() ) code |= it->getStatusCode();
        status->setValue(id, code);

      }

      if ( strncmp(module.c_str(), "EE+", 3) == 0 || strncmp(module.c_str(), "EE-", 3) == 0 ) {

#if 0
        int jx, jy;
        std::string word;
        stream >> jx >> jy >> word;
#else
        int index;
        std::string word;
        stream >> index >> word;
        int jx = index/1000;
        int jy = index%1000;
#endif

        int ism = atoi( module.substr(2, module.size()-2).c_str() );
        if( ism == -99 ) ism = -1;
        if( ism == +99 ) ism = +1;
        switch ( ism ) {
          case +7: ism =  1; break;
          case +8: ism =  2; break;
          case +9: ism =  3; break;
          case +1: ism =  4; break;
          case +2: ism =  5; break;
          case +3: ism =  6; break;
          case +4: ism =  7; break;
          case +5: ism =  8; break;
          case +6: ism =  9; break;
          case -7: ism = 10; break;
          case -8: ism = 11; break;
          case -9: ism = 12; break;
          case -1: ism = 13; break;
          case -2: ism = 14; break;
          case -3: ism = 15; break;
          case -4: ism = 16; break;
          case -5: ism = 17; break;
          case -6: ism = 18; break;
        }

        EEDetId id(jx, jy, (ism>=1&&ism<=9)?-1:+1, EEDetId::XYMODE);
        int code = -1;
        for (unsigned int i=0; i<dictionary.size(); i++) {
          if ( strcmp(word.c_str(), dictionary[i].desc) == 0 ) {
            code = dictionary[i].code;
          }
        }
        if ( code == -1 ) {
          std::cout << " --> not found in the dictionary: " << word << std::endl;
          continue;
        }

        int hashedIndex = id.hashedIndex();
        if ( code !=0 ) std::cout << module << " hashedIndex " << hashedIndex << " status " <<  code << std::endl;
        EcalDQMChannelStatus::const_iterator it = status->find(id);
        if ( it != status->end() ) code |= it->getStatusCode();
        status->setValue(id, code);

      }

    } else {

      std:: cout << "--> skipped line #" << ii << " : " << token << std::endl;

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

  int ii = 0;
  while ( fgets(line, 255, ifile) ) {

    std::string token = "";

    std::stringstream stream;

    ii++;
    stream << line;
    stream >> token;

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

