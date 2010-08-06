// $Id: Masks.cc,v 1.8 2010/08/05 20:25:46 dellaric Exp $

/*!
  \file Masks.cc
  \brief channel masking
  \author G. Della Ricca
  \version $Revision: 1.8 $
  \date $Date: 2010/08/05 20:25:46 $
*/

#include <sstream>
#include <iomanip>

#include "DQMServices/Core/interface/DQMStore.h"

#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>

#include <DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h>
#include <DataFormats/EcalDetId/interface/EcalScDetId.h>

#include "FWCore/Framework/interface/NoRecordException.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/EcalDQMChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalDQMTowerStatusRcd.h"

#include "DQM/EcalCommon/interface/UtilsClient.h"
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

  if ( setup.find( edm::eventsetup::EventSetupRecordKey::makeKey< EcalDQMChannelStatusRcd >() ) ) {
    edm::ESHandle< EcalDQMChannelStatus > handle;
    setup.get< EcalDQMChannelStatusRcd >().get(handle);
    if ( handle.isValid() ) Masks::channelStatus = handle.product();
  }

  if ( setup.find( edm::eventsetup::EventSetupRecordKey::makeKey< EcalDQMTowerStatusRcd >() ) ) {
    edm::ESHandle< EcalDQMTowerStatus > handle;
    setup.get< EcalDQMTowerStatusRcd >().get(handle);
    if ( handle.isValid() ) Masks::towerStatus = handle.product();
  }

  if ( verbose ) std::cout << "done." << std::endl;

}

//-------------------------------------------------------------------------

bool Masks::maskChannel( int ism, int ix, int iy, uint32_t bits, const EcalSubdetector subdet ) throw( std::runtime_error ) {

  bool mask = false;

  if ( subdet == EcalBarrel ) {

    int iex = (ism>=1&&ism<=18) ? -ix : +ix;
    int ipx = (ism>=1&&ism<=18) ? iy+20*(ism-1) : 1+(20-iy)+20*(ism-19);

    if ( EBDetId::validDetId(iex, ipx) ) {
      EBDetId id(iex, ipx);
      if ( Masks::channelStatus ) {
        EcalDQMChannelStatus::const_iterator it = Masks::channelStatus->find( id.rawId() );
        if ( it != Masks::channelStatus->end() ) mask |= it->getStatusCode() & bits;
      }
      if ( towerStatus ) {
        EcalDQMTowerStatus::const_iterator it = Masks::towerStatus->find( id.tower().rawId() );
        if ( it != Masks::towerStatus->end() ) mask |= it->getStatusCode() & bits;
      }
    }

  } else if ( subdet == EcalEndcap ) {

    int jx = ix + Numbers::ix0EE(ism);
    int jy = iy + Numbers::iy0EE(ism);

    if ( ism >= 1 && ism <= 9 ) jx = 101 - jx;

    if ( Numbers::validEE(ism, jx, jy) ) {
      EEDetId id(jx, jy, (ism>=1&&ism<=9)?-1:+1, EEDetId::XYMODE);
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
    throw( std::runtime_error( s.str() ) );

  }

  return ( mask );

}

//-------------------------------------------------------------------------

bool Masks::maskPn( int ism, int ix, uint32_t bits, const EcalSubdetector subdet ) throw( std::runtime_error ) {

  bool mask = false;

  if ( subdet == EcalBarrel ) {

    // EB-03
    if ( ism ==  3 && ix ==  1 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;

    // EB-07
    if ( ism ==  7 && ix ==  4 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;

    // EB+07
    if ( ism == 25 && ix ==  4 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;

    // EB+12
    if ( ism == 30 && ix ==  9 && (bits & (1 << EcalDQMStatusHelper::LASER_MEAN_ERROR)) ) mask = true;

  } else if ( subdet == EcalEndcap ) {

    // EE-02
    if ( ism == 5 && ix ==  3 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 5 && ix ==  3 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 5 && ix == 10 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 5 && ix == 10 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR)) ) mask = true;

    // EE-03
    if ( ism == 6 && ix ==  5 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 6 && ix ==  5 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR)) ) mask = true;

    // EE-07
    if ( ism == 1 && ix ==  4 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 1 && ix ==  4 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 1 && ix == 10 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 1 && ix == 10 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR)) ) mask = true;

    // EE-08
    if ( ism == 2 && ix ==  1 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 2 && ix ==  1 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 2 && ix ==  4 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 2 && ix ==  4 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 2 && ix == 10 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR)) ) mask = true;
    if ( ism == 2 && ix == 10 && (bits & (1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR)) ) mask = true;

  } else {

    std::ostringstream s;
    s << "Invalid subdetector: subdet = " << subdet;
    throw( std::runtime_error( s.str() ) );

  }

  return ( mask );

}

//-------------------------------------------------------------------------

