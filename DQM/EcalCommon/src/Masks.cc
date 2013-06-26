// $Id: Masks.cc,v 1.18 2012/04/27 13:46:04 yiiyama Exp $

/*!
  \file Masks.cc
  \brief channel masking
  \author G. Della Ricca
  \version $Revision: 1.18 $
  \date $Date: 2012/04/27 13:46:04 $
*/

#include <sstream>
#include <iomanip>

#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/EcalDQMChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalDQMTowerStatusRcd.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalCommon/interface/Masks.h"

//-------------------------------------------------------------------------

const EcalDQMChannelStatus* Masks::channelStatus = 0;
const EcalDQMTowerStatus* Masks::towerStatus = 0;

bool Masks::init = false;

//-------------------------------------------------------------------------

void Masks::initMasking( const edm::EventSetup& setup, bool verbose ) {

  if ( Masks::init ) return;

  if ( verbose ) std::cout << "Initializing EcalDQMChannelStatus and EcalDQMTowerStatus ..." << std::endl;

  Masks::init = true;

  if ( setup.find( edm::eventsetup::EventSetupRecordKey::makeKey<EcalDQMChannelStatusRcd>() ) ) {
    edm::ESHandle<EcalDQMChannelStatus> handle;
    setup.get<EcalDQMChannelStatusRcd>().get(handle);
    if ( handle.isValid() ) Masks::channelStatus = handle.product();
  }

  if ( setup.find( edm::eventsetup::EventSetupRecordKey::makeKey<EcalDQMTowerStatusRcd>() ) ) {
    edm::ESHandle<EcalDQMTowerStatus> handle;
    setup.get<EcalDQMTowerStatusRcd>().get(handle);
    if ( handle.isValid() ) Masks::towerStatus = handle.product();
  }

  if ( verbose ) std::cout << "done." << std::endl;

}

//-------------------------------------------------------------------------

bool Masks::maskChannel( int ism, int i1, int i2, uint32_t bits, const EcalSubdetector subdet ) throw( cms::Exception ) {

  bool mask = false;

  if ( subdet == EcalBarrel ) {

    int jsm = Numbers::iSM(ism, EcalBarrel);
    int ic = 20*(i1-1)+(i2-1)+1;

    EBDetId id(jsm, ic, EBDetId::SMCRYSTALMODE);
    if ( Masks::channelStatus ) {
      EcalDQMChannelStatus::const_iterator it = Masks::channelStatus->find( id.rawId() );
      if ( it != Masks::channelStatus->end() ) mask |= it->getStatusCode() & bits;
    }
    if ( Masks::towerStatus ) {
      EcalDQMTowerStatus::const_iterator it = Masks::towerStatus->find( id.tower().rawId() );
      if ( it != Masks::towerStatus->end() ) mask |= it->getStatusCode() & bits;
    }

  } else if ( subdet == EcalEndcap ) {

    int jx = i1 + Numbers::ix0EE(ism);
    int jy = i2 + Numbers::iy0EE(ism);

    if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

    if ( Numbers::validEE(ism, jx, jy) ) {
      EEDetId id(jx, jy, (ism>=1&&ism<=9)?-1:+1);
      if ( Masks::channelStatus ) {
        EcalDQMChannelStatus::const_iterator it = Masks::channelStatus->find( id.rawId() );
        if ( it != Masks::channelStatus->end() ) mask |= it->getStatusCode() & bits;
      }
      if ( Masks::towerStatus ) {
        EcalDQMTowerStatus::const_iterator it = Masks::towerStatus->find( id.sc().rawId() );
        if ( it != Masks::towerStatus->end() ) mask |= it->getStatusCode() & bits;
      }
   }

  } else {

    std::ostringstream s;
    s << "Invalid subdetector: subdet = " << subdet;
    throw( cms::Exception( s.str() ) );

  }

  return ( mask );

}

//-------------------------------------------------------------------------

bool Masks::maskPn( int ism, int i1, uint32_t bits, const EcalSubdetector subdet ) throw( cms::Exception ) {

  bool mask = false;

  if ( subdet == EcalBarrel ) {

    // EB-03
    if ( ism ==  3 && i1 ==  1 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;
    if ( ism ==  3 && i1 ==  1 && (bits & (1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR)) ) mask = true;

    // EB-07
    if ( ism ==  7 && i1 ==  4 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;
    if ( ism ==  7 && i1 ==  4 && (bits & (1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR)) ) mask = true;

    // EB-15
    if ( ism == 15 && i1 ==  6 && (bits & (1 << EcalDQMStatusHelper::TT_SIZE_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  6 && (bits & (1 << EcalDQMStatusHelper::STATUS_FLAG_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  6 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  6 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  6 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  7 && (bits & (1 << EcalDQMStatusHelper::TT_SIZE_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  7 && (bits & (1 << EcalDQMStatusHelper::STATUS_FLAG_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  7 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  7 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  7 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  8 && (bits & (1 << EcalDQMStatusHelper::TT_SIZE_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  8 && (bits & (1 << EcalDQMStatusHelper::STATUS_FLAG_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  8 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  8 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  8 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  9 && (bits & (1 << EcalDQMStatusHelper::TT_SIZE_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  9 && (bits & (1 << EcalDQMStatusHelper::STATUS_FLAG_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  9 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  9 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  9 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;
    if ( ism == 15 && i1 == 10 && (bits & (1 << EcalDQMStatusHelper::TT_SIZE_ERROR)) ) mask = true;
    if ( ism == 15 && i1 == 10 && (bits & (1 << EcalDQMStatusHelper::STATUS_FLAG_ERROR)) ) mask = true;
    if ( ism == 15 && i1 == 10 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 15 && i1 == 10 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 15 && i1 == 10 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;

    // EB+06
    if ( ism == 24 &&             (bits & (1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR)) ) mask = true;

    // EB+07
    if ( ism == 25 && i1 ==  4 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;
    if ( ism == 25 && i1 ==  4 && (bits & (1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR)) ) mask = true;

    // EB+12
    if ( ism == 30 && i1 ==  9 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;
    if ( ism == 30 && i1 ==  9 && (bits & (1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR)) ) mask = true;

    // EB+15
    if ( ism == 15 && i1 ==  3 && (bits & (1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  4 && (bits & (1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 15 && i1 ==  5 && (bits & (1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR)) ) mask = true;

  } else if ( subdet == EcalEndcap ) {

    // EE-02
    if ( ism == 5 && i1 ==  3 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;
    if ( ism == 5 && i1 ==  3 && (bits & (1 << EcalDQMStatusHelper::LED_MEAN_ERROR)) ) mask = true;
    if ( ism == 5 && i1 == 10 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;
    if ( ism == 5 && i1 == 10 && (bits & (1 << EcalDQMStatusHelper::LED_MEAN_ERROR)) ) mask = true;

    // EE-03
    if ( ism == 6 && i1 ==  5 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;
    if ( ism == 6 && i1 ==  5 && (bits & (1 << EcalDQMStatusHelper::LED_MEAN_ERROR)) ) mask = true;

    // EE-07
    if ( ism == 1 && i1 ==  4 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;
    if ( ism == 1 && i1 ==  4 && (bits & (1 << EcalDQMStatusHelper::LED_MEAN_ERROR)) ) mask = true;
    if ( ism == 1 && i1 == 10 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;
    if ( ism == 1 && i1 == 10 && (bits & (1 << EcalDQMStatusHelper::LED_MEAN_ERROR)) ) mask = true;

    // EE-08
    if ( ism == 2 && i1 ==  1 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;
    if ( ism == 2 && i1 ==  1 && (bits & (1 << EcalDQMStatusHelper::LED_MEAN_ERROR)) ) mask = true;
    if ( ism == 2 && i1 ==  4 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;
    if ( ism == 2 && i1 ==  4 && (bits & (1 << EcalDQMStatusHelper::LED_MEAN_ERROR)) ) mask = true;
    if ( ism == 2 && i1 ==  4 && (bits & (1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 2 && i1 == 10 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;
    if ( ism == 2 && i1 == 10 && (bits & (1 << EcalDQMStatusHelper::LED_MEAN_ERROR)) ) mask = true;

  } else {

    std::ostringstream s;
    s << "Invalid subdetector: subdet = " << subdet;
    throw( cms::Exception( s.str() ) );

  }

  return ( mask );

}

//-------------------------------------------------------------------------

