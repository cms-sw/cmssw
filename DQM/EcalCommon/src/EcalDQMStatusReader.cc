/*
 * \file EcalDQMStatusReader.cc
 *
 * $Date: 2010/08/08 20:03:00 $
 * $Revision: 1.7 $
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
  for ( int ie=-EBDetId::MAX_IETA; ie<=EBDetId::MAX_IETA; ie++ ) {
  if ( ie==0 ) continue;
    for ( int ip=EBDetId::MIN_IPHI; ip<=EBDetId::MAX_IPHI; ip++ ) {
      if ( EBDetId::validDetId(ie, ip) ) {
        EBDetId id(ie, ip, EBDetId::ETAPHIMODE);
        if ( channelStatus ) {
          EcalDQMChannelStatus::const_iterator it = channelStatus->find( id.rawId() );
          if ( it != channelStatus->end() ) {
            if ( it->getStatusCode() != 0 ) {
              std::cout << "# EB:channel, ic=" << id.ic() << " hi=" << id.hashedIndex() << " status=" << it->getStatusCode() << std::endl;
              std::vector<EcalDQMStatusDictionary::codeDef> codes;
              EcalDQMStatusDictionary::getCodes( codes, it->getStatusCode() );
              for ( unsigned int i=0; i<codes.size(); i++ ) {
                std::cout << "Crystal " << Numbers::sEB(id.ism()) << " " << id.ic() << " " << codes[i].desc << std::endl;
              }
            }
          }
        }
        if ( towerStatus ) {
          EcalDQMTowerStatus::const_iterator it = towerStatus->find( id.tower().rawId() );
          if ( it != towerStatus->end() ) {
            if ( it->getStatusCode() != 0 ) {
              std::cout << "# EB:tower, ic=" << id.ic() << " hi=" << id.hashedIndex() << " status=" << it->getStatusCode() << std::endl;
              std::vector<EcalDQMStatusDictionary::codeDef> codes;
              EcalDQMStatusDictionary::getCodes( codes, it->getStatusCode() );
              for ( unsigned int i=0; i<codes.size(); i++ ) {
                std::cout << "TT " << Numbers::sEB(Numbers::iSM(id.ism(), EcalBarrel)) << " " << id.tower().iTT() << " " << codes[i].desc << std::endl;
              }
            }
          }
        }
      }
    }
  }

  // endcap
  for ( int ix=EEDetId::IX_MIN; ix<=EEDetId::IX_MAX; ix++ ) {
    for ( int iy=EEDetId::IY_MIN; iy<=EEDetId::IY_MAX; iy++ ) {
      if ( EEDetId::validDetId(ix, iy, +1) ) {
        EEDetId id(ix, iy, +1, EEDetId::XYMODE);
        if ( channelStatus ) {
          EcalDQMChannelStatus::const_iterator it = channelStatus->find( id.rawId() );
          if ( it != channelStatus->end() ) {
            if ( it->getStatusCode() != 0 ) {
              std::cout << "# EE:channel, " << ix << " " << iy << " hi=" << id.hashedIndex() << " " << it->getStatusCode() << std::endl;
              std::vector<EcalDQMStatusDictionary::codeDef> codes;
              EcalDQMStatusDictionary::getCodes( codes, it->getStatusCode() );
              for ( unsigned int i=0; i<codes.size(); i++ ) {
                std::cout << "Crystal " << Numbers::sEE(Numbers::iSM(id)) << " " << Numbers::indexEE(Numbers::iSM(id), ix, iy) << " " << codes[i].desc << std::endl;
              }
            }
          }
        }
        if ( towerStatus ) {
          EcalDQMTowerStatus::const_iterator it = towerStatus->find( id.sc().rawId() );
          if ( it != towerStatus->end() ) {
            if ( it->getStatusCode() != 0 ) {
              std::cout << "# EE:tower, " << ix << " " << iy << " hi=" << id.hashedIndex() << " " << it->getStatusCode() << std::endl;
              std::vector<EcalDQMStatusDictionary::codeDef> codes;
              EcalDQMStatusDictionary::getCodes( codes, it->getStatusCode() );
              for ( unsigned int i=0; i<codes.size(); i++ ) {
                std::cout << "TT " << Numbers::sEE(Numbers::iSM(id)) << " " << Numbers::iSC(Numbers::iSM(id), EcalEndcap, ix, iy) << " " << codes[i].desc << std::endl;
              }
            }
          }
        }
      }
      if ( EEDetId::validDetId(ix, iy, -1) ) {
        EEDetId id(ix, iy, -1, EEDetId::XYMODE);
        if ( channelStatus ) {
          EcalDQMChannelStatus::const_iterator it = channelStatus->find( id.rawId() );
          if ( it != channelStatus->end() ) {
            if ( it->getStatusCode() != 0 ) {
              std::cout << "# EE:channel, " << ix << " " << iy << " hi=" << id.hashedIndex() << " " << it->getStatusCode() << std::endl;
              std::vector<EcalDQMStatusDictionary::codeDef> codes;
              EcalDQMStatusDictionary::getCodes( codes, it->getStatusCode() );
              for ( unsigned int i=0; i<codes.size(); i++ ) {
                std::cout << "Crystal " << Numbers::sEE(Numbers::iSM(id)) << " " << Numbers::indexEE(Numbers::iSM(id), ix, iy) << " " << codes[i].desc << std::endl;
              }
            }
          }
        }
        if ( towerStatus ) {
          EcalDQMTowerStatus::const_iterator it = towerStatus->find( id.sc().rawId() );
          if ( it != towerStatus->end() ) {
            if ( it->getStatusCode() != 0 ) {
              std::cout << "# EE:tower, " << ix << " " << iy << " hi=" << id.hashedIndex() << " " << it->getStatusCode() << std::endl;
              std::vector<EcalDQMStatusDictionary::codeDef> codes;
              EcalDQMStatusDictionary::getCodes( codes, it->getStatusCode() );
              for ( unsigned int i=0; i<codes.size(); i++ ) {
                std::cout << "TT " << Numbers::sEE(Numbers::iSM(id)) << " " << Numbers::iSC(Numbers::iSM(id), EcalEndcap, ix, iy) << " " << codes[i].desc << std::endl;
              }
            }
          }
        }
      }
    }
  }

}

void EcalDQMStatusReader::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EcalDQMStatusReader::beginJob() {

}

void EcalDQMStatusReader::endJob() {

}

void EcalDQMStatusReader::analyze(const edm::Event& e, const edm::EventSetup& c) {

}

EcalDQMStatusReader::~EcalDQMStatusReader() {

}

