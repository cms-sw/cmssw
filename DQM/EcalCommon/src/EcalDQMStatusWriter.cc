/*
 * \file EcalDQMStatusWriter.cc
 *
 * $Date: 2010/08/30 13:14:08 $
 * $Revision: 1.19 $
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

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"

#include "DQM/EcalCommon/interface/EcalDQMStatusDictionary.h"

#include "DQM/EcalCommon/interface/EcalDQMStatusWriter.h"

EcalDQMStatusWriter::EcalDQMStatusWriter(const edm::ParameterSet& ps) {

  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters toPut = ps.getParameter<Parameters>("toPut");

  for ( Parameters::iterator itToPut=toPut.begin(); itToPut!=toPut.end(); itToPut++ ) {
    inpFileName_.push_back(itToPut->getUntrackedParameter<std::string>("inputFile"));
    objectName_.push_back(itToPut->getUntrackedParameter<std::string>("conditionType"));
    since_.push_back(itToPut->getUntrackedParameter<unsigned int>("since"));
  }

}

void EcalDQMStatusWriter::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, verbose_);

  edm::Service<cond::service::PoolDBOutputService> dbservice;
  if ( !dbservice.isAvailable() ){
    std::cout << "PoolDBOutputService is unavailable" << std::endl;
    return;
  }

  for ( unsigned int i=0; i<objectName_.size(); i++ ) {

    bool toAppend;
    cond::Time_t newTime;

    if ( dbservice->isNewTagRequest( objectName_[i]+std::string("Rcd") ) ) {
      // This is the first object for this tag.
      // Append mode should be off.
      // newTime is the end of this new objects IOV.
      toAppend = false;
      newTime = dbservice->beginOfTime();
    } else {
      // There should already be an object in the DB for this tag.
      // Append IOV mode should be on.
      // newTime is the beginning of this new objects IOV.
      toAppend = true;
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

EcalDQMChannelStatus* EcalDQMStatusWriter::readEcalDQMChannelStatusFromFile(const char* inputFile) {

  EcalDQMChannelStatus* status = new EcalDQMChannelStatus();

  std::vector<EcalDQMStatusDictionary::codeDef> dictionary;
  EcalDQMStatusDictionary::getDictionary( dictionary );

  // barrel
  for ( int ism=1; ism<=36; ism++ ) {
    int jsm = Numbers::iSM(ism, EcalBarrel);
    for ( int ic=1; ic<=1700; ic++ ) {
      EBDetId id(jsm, ic, EBDetId::SMCRYSTALMODE);
      status->setValue(id, 0);
    }
  }

  // endcap
  for ( int ix=1; ix<=100; ix++ ) {
    for ( int iy=1; iy<=100; iy++ ) {
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

    std::stringstream stream;

    ii++;
    std::string key;
    stream << line;
    if ( verbose_ ) std::cout << line;
    stream >> key;

    if ( key.size() == 0 || strcmp(key.c_str(), " ") == 0 || strncmp(key.c_str(), "#", 1) == 0 ) {

      // skip comments

    } else if ( strcmp(key.c_str(), "EB") == 0 ) {

      int index;
      uint32_t code;
      stream >> index >> code;

      EBDetId id = EBDetId::unhashIndex(index);
      code = convert(code);

      EcalDQMChannelStatus::const_iterator it = status->find(id);
      if ( it != status->end() ) code |= it->getStatusCode();
      status->setValue(id, code);

    } else if ( strcmp(key.c_str(), "EE") == 0 ) {

      int ix, iy, iz;
      uint32_t code;
      stream >> ix >> iy >> iz >> code;

      EEDetId id(ix, iy, iz);
      code = convert(code);

      EcalDQMChannelStatus::const_iterator it = status->find(id);
      if ( it != status->end() ) code |= it->getStatusCode();
      status->setValue(id, code);

    } else if ( strcmp(key.c_str(), "Crystal") == 0 ) {

      std::string module;
      stream >> module;

      if ( strncmp(module.c_str(), "EB+", 3) == 0 || strncmp(module.c_str(), "EB-", 3) == 0 ) {

#if 0
        int ie, ip;
        std::string token;
        stream >> ie >> ip >> token;

        int sm = atoi( module.substr(2, module.size()-2).c_str() );

        int ism = (sm>=1&&sm<=18) ? sm+18 : -sm;
        int jsm = Numbers::iSM(ism, EcalBarrel);
        int ic = 20*(ie-1)+(ip-1)+1;
#else
        int ic;
        std::string token;
        stream >> ic >> token;

        int sm = atoi( module.substr(2, module.size()-2).c_str() );

        int ism = (sm>=1&&sm<=18) ? sm+18 : -sm;
        int jsm = Numbers::iSM(ism, EcalBarrel);
#endif

        EBDetId id(jsm, ic, EBDetId::SMCRYSTALMODE);
        uint32_t code = 0;
        for ( unsigned int i=0; i<dictionary.size(); i++ ) {
          if ( strcmp(token.c_str(), dictionary[i].desc) == 0 ) {
            code = dictionary[i].code;
          }
        }
        if ( code == 0 ) {
          std::cout << " --> not found in the dictionary: " << token << std::endl;
          continue;
        }

        EcalDQMChannelStatus::const_iterator it = status->find(id);
        if ( it != status->end() ) code |= it->getStatusCode();
        status->setValue(id, code);

      } else if ( strncmp(module.c_str(), "EE+", 3) == 0 || strncmp(module.c_str(), "EE-", 3) == 0 ) {

#if 0
        int jx, jy;
        std::string token;
        stream >> jx >> jy >> token;
#else
        int index;
        std::string token;
        stream >> index >> token;
        int jx = index/1000;
        int jy = index%1000;
#endif

        int sm = atoi( module.substr(2, module.size()-2).c_str() );

        int ism = 0;
        if( sm == -99 ) sm = -1;
        if( sm == +99 ) sm = +1;
        switch ( sm ) {
          case -7: ism =  1; break;
          case -8: ism =  2; break;
          case -9: ism =  3; break;
          case -1: ism =  4; break;
          case -2: ism =  5; break;
          case -3: ism =  6; break;
          case -4: ism =  7; break;
          case -5: ism =  8; break;
          case -6: ism =  9; break;
          case +7: ism = 10; break;
          case +8: ism = 11; break;
          case +9: ism = 12; break;
          case +1: ism = 13; break;
          case +2: ism = 14; break;
          case +3: ism = 15; break;
          case +4: ism = 16; break;
          case +5: ism = 17; break;
          case +6: ism = 18; break;
        }

        EEDetId id(jx, jy, (ism>=1&&ism<=9)?-1:+1);
        uint32_t code = 0;
        for ( unsigned int i=0; i<dictionary.size(); i++ ) {
          if ( strcmp(token.c_str(), dictionary[i].desc) == 0 ) {
            code = dictionary[i].code;
          }
        }
        if ( code == 0 ) {
          std::cout << " --> not found in the dictionary: " << token << std::endl;
          continue;
        }

        EcalDQMChannelStatus::const_iterator it = status->find(id);
        if ( it != status->end() ) code |= it->getStatusCode();
        status->setValue(id, code);

      } else {

        std:: cout << "--> unknown token at line #" << ii << " : " << line;

      }

    } else if ( strcmp(key.c_str(), "TT") == 0 ) {

      // see readEcalDQMTowerStatusFromFile

    } else if ( strcmp(key.c_str(), "PN") == 0 || strcmp(key.c_str(), "MemCh") == 0 || strcmp(key.c_str(), "MemTT") == 0 ) {

      std::cout << "--> unsupported key at line #" << ii << " : " << line;

    } else {

      std:: cout << "--> skipped line #" << ii << " : " << line;

    }

  }

  fclose(ifile);

  return status;

}

EcalDQMTowerStatus* EcalDQMStatusWriter::readEcalDQMTowerStatusFromFile(const char* inputFile) {

  EcalDQMTowerStatus* status = new EcalDQMTowerStatus();

  std::vector<EcalDQMStatusDictionary::codeDef> dictionary;
  EcalDQMStatusDictionary::getDictionary( dictionary );

  // barrel
  for ( int ix=1; ix<=17; ix++ ) {
    for ( int iy=1; iy<=72; iy++ ) {
      if ( EcalTrigTowerDetId::validDetId(+1, EcalBarrel, ix, iy) ) {
        EcalTrigTowerDetId id(+1, EcalBarrel, ix, iy);
        status->setValue(id, 0);
      }
      if ( EcalTrigTowerDetId::validDetId(-1, EcalBarrel, ix, iy) ) {
        EcalTrigTowerDetId id(-1, EcalBarrel, ix, iy);
        status->setValue(id, 0);
      }
    }
  }
  
  // endcap
  for ( int ix=1; ix<=20; ix++ ) {
    for ( int iy=1; iy<=20; iy++ ) {
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

  std::cout << "Reading tower status from file " << inputFile << std::endl;
  FILE *ifile = fopen( inputFile ,"r" );

  if ( !ifile ) throw cms::Exception ("Cannot open file") ;

  char line[256];

  int ii = 0;
  while ( fgets(line, 255, ifile) ) {

    std::stringstream stream;

    ii++;
    std::string key;
    stream << line;
    if ( verbose_ ) std::cout << line;
    stream >> key;

    if ( key.size() == 0 || strcmp(key.c_str(), " ") == 0 || strncmp(key.c_str(), "#", 1) == 0 ) {

      // skip comments

    } else if ( strcmp(key.c_str(), "EB") == 0 ) {

      // see readEcalDQMChannelStatusFromFile

    } else if ( strcmp(key.c_str(), "EE") == 0 ) {

      // see readEcalDQMChannelStatusFromFile

    } else if ( strcmp(key.c_str(), "Crystal") == 0 ) {

      // see readEcalDQMChannelStatusFromFile

    } else if ( strcmp(key.c_str(), "TT") == 0 ) {

      std::string module;
      stream >> module;

      if ( strncmp(module.c_str(), "EB+", 3) == 0 || strncmp(module.c_str(), "EB-", 3) == 0 ) {

        int itt;
        std::string token;
        stream >> itt >> token;

        if ( itt >= 1 && itt <= 68 ) {

          int sm = atoi( module.substr(2, module.size()-2).c_str() );

          int iet = (itt-1)/4+1;
          int ipt = (itt-1)%4+1;

          if ( sm<0 ) {
            ipt = ipt+(std::abs(sm)-1)*4-2;
            if ( ipt < 1 ) ipt = ipt+72;
            if ( ipt > 72 ) ipt = ipt-72;
          } else {
            ipt = (5-ipt)+(std::abs(sm)-1)*4-2;
            if ( ipt < 1 ) ipt = ipt+72;
            if ( ipt > 72 ) ipt = ipt-72;
          }

          EcalTrigTowerDetId id((sm<0)?-1:+1, EcalBarrel, iet, ipt);
          uint32_t code = 0;
          for ( unsigned int i=0; i<dictionary.size(); i++ ) {
            if ( strcmp(token.c_str(), dictionary[i].desc) == 0 ) {
              code = dictionary[i].code;
            }
          }
          if ( code == 0 ) {
            std::cout << " --> not found in the dictionary: " << token << std::endl;
            continue;
          }

          EcalDQMTowerStatus::const_iterator it = status->find(id);
          if ( it != status->end() ) code |= it->getStatusCode();
          status->setValue(id, code);

        } else {

          std::cout << "--> unsupported configuration at line #" << ii << " : " << line;

        }

      } else if ( strncmp(module.c_str(), "EE+", 3) == 0 || strncmp(module.c_str(), "EE-", 3) == 0 ) {

        int isc;
        std::string token;
        stream >> isc >> token;

        if ( isc >= 1 && isc <= 68 ) {

          int sm = atoi( module.substr(2, module.size()-2).c_str() );

          int ism = 0;
          switch ( sm ) {
            case -7: ism =  1; break;
            case -8: ism =  2; break;
            case -9: ism =  3; break;
            case -1: ism =  4; break;
            case -2: ism =  5; break;
            case -3: ism =  6; break;
            case -4: ism =  7; break;
            case -5: ism =  8; break;
            case -6: ism =  9; break;
            case +7: ism = 10; break;
            case +8: ism = 11; break;
            case +9: ism = 12; break;
            case +1: ism = 13; break;
            case +2: ism = 14; break;
            case +3: ism = 15; break;
            case +4: ism = 16; break;
            case +5: ism = 17; break;
            case +6: ism = 18; break;
          }

          int idcc = (ism>=1&&ism<=9) ? ism : ism-9+45;

          std::vector<DetId>* crystals = Numbers::crystals(idcc, isc);

          for ( unsigned int i=0; i<crystals->size(); i++ ) {

            EcalScDetId id = ((EEDetId) (*crystals)[i]).sc();
            uint32_t code = 0;
            for ( unsigned int i=0; i<dictionary.size(); i++ ) {
              if ( strcmp(token.c_str(), dictionary[i].desc) == 0 ) {
                code = dictionary[i].code;
              }
            }
            if ( code == 0 ) {
              std::cout << " --> not found in the dictionary: " << token << std::endl;
              continue;
            }

            EcalDQMTowerStatus::const_iterator it = status->find(id);
            if ( it != status->end() ) code |= it->getStatusCode();
            status->setValue(id, code);

          }

        } else {

          std::cout << "--> unsupported configuration at line #" << ii << " : " << line;

        }

      } else {

        std:: cout << "--> unknown token at line #" << ii << " : " << line;

      }

    } else if ( strcmp(key.c_str(), "PN") == 0 || strcmp(key.c_str(), "MemCh") == 0 || strcmp(key.c_str(), "MemTT") == 0 ) {

      std::cout << "--> unsupported key at line #" << ii << " : " << line;

    } else {

      std:: cout << "--> skipped line #" << ii << " : " << line;

    }

  }

  fclose(ifile);

  return status;

}

uint32_t EcalDQMStatusWriter::convert(uint32_t chStatus) {

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

