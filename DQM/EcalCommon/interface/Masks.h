// $Id: Masks.h,v 1.2 2010/08/05 11:35:07 dellaric Exp $

/*!
  \file Masks.h
  \brief Some "id" conversions
  \author B. Gobbo 
  \version $Revision: 1.2 $
  \date $Date: 2010/08/05 11:35:07 $
*/

#ifndef MASKS_H
#define MASKS_H

#include <string>
#include <stdexcept>
#include <vector>

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

class Masks {

 public:

  static void initMasking( const edm::EventSetup& setup, bool verbose = false );

  static bool maskChannel( int ism, int ix, int iy, uint32_t bits, const EcalSubdetector subdet ) throw( std::runtime_error );

  static bool maskPn( int ism, int ix, uint32_t bits, const EcalSubdetector subdet ) throw( std::runtime_error );

private:

  static bool init;

  static const EcalDQMChannelStatus* channelStatus;
  static const EcalDQMTowerStatus* towerStatus;

};

#endif // MASKS_H
