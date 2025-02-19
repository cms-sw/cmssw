/*
 * \file EcalDQMStatusReader.cc
 *
 * $Date: 2010/08/09 17:47:32 $
 * $Revision: 1.11 $
 * \author G. Della Ricca
 *
*/

#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <time.h>
#include <unistd.h>

#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/EcalDQMChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalDQMTowerStatusRcd.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalCommon/interface/EcalDQMStatusDictionary.h"

#include "DQM/EcalCommon/interface/EcalDQMStatusReader.h"

EcalDQMStatusReader::EcalDQMStatusReader(const edm::ParameterSet& ps) {

  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

}

void EcalDQMStatusReader::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, verbose_);

  std::vector<EcalDQMStatusDictionary::codeDef> dictionary;
  EcalDQMStatusDictionary::getDictionary( dictionary );

  const EcalDQMChannelStatus* channelStatus = 0;
  if ( c.find( edm::eventsetup::EventSetupRecordKey::makeKey< EcalDQMChannelStatusRcd >() ) ) {
    edm::ESHandle< EcalDQMChannelStatus > handle;
    c.get< EcalDQMChannelStatusRcd >().get(handle);
    if ( handle.isValid() ) channelStatus = handle.product();
  }

  const EcalDQMTowerStatus* towerStatus = 0;
  if ( c.find( edm::eventsetup::EventSetupRecordKey::makeKey< EcalDQMTowerStatusRcd >() ) ) {
    edm::ESHandle< EcalDQMTowerStatus > handle;
    c.get< EcalDQMTowerStatusRcd >().get(handle);
    if ( handle.isValid() ) towerStatus = handle.product();
  }

  // barrel
  for ( int ism=1; ism<=36; ism++ ) {
    for ( int ic=1; ic<=1700; ic++ ) {
      int jsm = Numbers::iSM(ism, EcalBarrel);
      EBDetId id(jsm, ic, EBDetId::SMCRYSTALMODE);
      if ( channelStatus ) {
        EcalDQMChannelStatus::const_iterator it = channelStatus->find( id.rawId() );
        if ( it != channelStatus->end() ) {
          if ( it->getStatusCode() != 0 ) {
            if ( verbose_ ) std::cout << "# EB:channel, ic=" << ic << " hi=" << id.hashedIndex() << " status=" << it->getStatusCode() << std::endl;
            std::vector<EcalDQMStatusDictionary::codeDef> codes;
            EcalDQMStatusDictionary::getCodes( codes, it->getStatusCode() );
            for ( unsigned int i=0; i<codes.size(); i++ ) {
              std::cout << "Crystal " << Numbers::sEB(ism) << " " << ic << " " << codes[i].desc << std::endl;
            }
          }
        }
      }
    }
  }

  for ( int ix=1; ix<=17; ix++ ) {
    for ( int iy=1; iy<=72; iy++ ) {
      if ( EcalTrigTowerDetId::validDetId(+1, EcalBarrel, ix, iy) ) {
        EcalTrigTowerDetId id(+1, EcalBarrel, ix, iy);
        if ( towerStatus ) {
          EcalDQMTowerStatus::const_iterator it = towerStatus->find( id.rawId() );
          if ( it != towerStatus->end() ) {
            if ( it->getStatusCode() != 0 ) {
              if ( verbose_ ) std::cout << "# EB:tower, tt=" << Numbers::iTT(id) << " hi=" << id.hashedIndex() << " status=" << it->getStatusCode() << std::endl;
              std::vector<EcalDQMStatusDictionary::codeDef> codes;
              EcalDQMStatusDictionary::getCodes( codes, it->getStatusCode() );
              for ( unsigned int i=0; i<codes.size(); i++ ) {
                std::cout << "TT " << Numbers::sEB(Numbers::iSM(id)) << " " << Numbers::iTT(id) << " " << codes[i].desc << std::endl;
              }
            }
          }
        }
      }
      if ( EcalTrigTowerDetId::validDetId(-1, EcalBarrel, ix, iy) ) {
        EcalTrigTowerDetId id(-1, EcalBarrel, ix, iy);
        if ( towerStatus ) {
          EcalDQMTowerStatus::const_iterator it = towerStatus->find( id.rawId() );
          if ( it != towerStatus->end() ) {
            if ( it->getStatusCode() != 0 ) {
              if ( verbose_ ) std::cout << "# EB:tower, tt=" << Numbers::iTT(id) << " hi=" << id.hashedIndex() << " status=" << it->getStatusCode() << std::endl;
              std::vector<EcalDQMStatusDictionary::codeDef> codes;
              EcalDQMStatusDictionary::getCodes( codes, it->getStatusCode() );
              for ( unsigned int i=0; i<codes.size(); i++ ) {
                std::cout << "TT " << Numbers::sEB(Numbers::iSM(id)) << " " << Numbers::iTT(id) << " " << codes[i].desc << std::endl;
              }
            }
          }
        }
      }
    }
  }

  // endcap
  for ( int ix=1; ix<=100; ix++ ) {
    for ( int iy=1; iy<=100; iy++ ) {
      if ( EEDetId::validDetId(ix, iy, +1) ) {
        EEDetId id(ix, iy, +1);
        if ( channelStatus ) {
          EcalDQMChannelStatus::const_iterator it = channelStatus->find( id.rawId() );
          if ( it != channelStatus->end() ) {
            if ( it->getStatusCode() != 0 ) {
              if ( verbose_ ) std::cout << "# EE:channel, " << Numbers::indexEE(Numbers::iSM(id), ix, iy) << " hi=" << id.hashedIndex() << " " << it->getStatusCode() << std::endl;
              std::vector<EcalDQMStatusDictionary::codeDef> codes;
              EcalDQMStatusDictionary::getCodes( codes, it->getStatusCode() );
              for ( unsigned int i=0; i<codes.size(); i++ ) {
                std::cout << "Crystal " << Numbers::sEE(Numbers::iSM(id)) << " " << Numbers::indexEE(Numbers::iSM(id), ix, iy) << " " << codes[i].desc << std::endl;
              }
            }
          }
        }
      }
      if ( EEDetId::validDetId(ix, iy, -1) ) {
        EEDetId id(ix, iy, -1);
        if ( channelStatus ) {
          EcalDQMChannelStatus::const_iterator it = channelStatus->find( id.rawId() );
          if ( it != channelStatus->end() ) {
            if ( it->getStatusCode() != 0 ) {
              if ( verbose_ ) std::cout << "# EE:channel, " << Numbers::indexEE(Numbers::iSM(id), ix, iy) << " hi=" << id.hashedIndex() << " " << it->getStatusCode() << std::endl;
              std::vector<EcalDQMStatusDictionary::codeDef> codes;
              EcalDQMStatusDictionary::getCodes( codes, it->getStatusCode() );
              for ( unsigned int i=0; i<codes.size(); i++ ) {
                std::cout << "Crystal " << Numbers::sEE(Numbers::iSM(id)) << " " << Numbers::indexEE(Numbers::iSM(id), ix, iy) << " " << codes[i].desc << std::endl;
              }
            }
          }
        }
      }
    }
  }

  for ( int ix=1; ix<=20; ix++ ) {
    for ( int iy=1; iy<=20; iy++ ) {
      if ( EcalScDetId::validDetId(ix, iy, +1) ) {
        EcalScDetId id(ix, iy, +1);
        if ( towerStatus ) {
          EcalDQMTowerStatus::const_iterator it = towerStatus->find( id.rawId() );
          if ( it != towerStatus->end() ) {
            if ( it->getStatusCode() != 0 ) {
              if ( verbose_ ) std::cout << "# EE:tower, " << Numbers::iSC(id) << " hi=" << id.hashedIndex() << " " << it->getStatusCode() << std::endl;
              std::vector<EcalDQMStatusDictionary::codeDef> codes;
              EcalDQMStatusDictionary::getCodes( codes, it->getStatusCode() );
              for ( unsigned int i=0; i<codes.size(); i++ ) {
                std::cout << "TT " << Numbers::sEE(Numbers::iSM(id)) << " " << Numbers::iSC(id) << " " << codes[i].desc << std::endl;
              }
            }
          }
        }
      }
      if ( EcalScDetId::validDetId(ix, iy, -1) ) {
        EcalScDetId id(ix, iy, -1);
        if ( towerStatus ) {
          EcalDQMTowerStatus::const_iterator it = towerStatus->find( id.rawId() );
          if ( it != towerStatus->end() ) {
            if ( it->getStatusCode() != 0 ) {
              if ( verbose_ ) std::cout << "# EE:tower, " << Numbers::iSC(id) << " hi=" << id.hashedIndex() << " " << it->getStatusCode() << std::endl;
              std::vector<EcalDQMStatusDictionary::codeDef> codes;
              EcalDQMStatusDictionary::getCodes( codes, it->getStatusCode() );
              for ( unsigned int i=0; i<codes.size(); i++ ) {
                std::cout << "TT " << Numbers::sEE(Numbers::iSM(id)) << " " << Numbers::iSC(id) << " " << codes[i].desc << std::endl;
              }
            }
          }
        }
      }
    }
  }

}

