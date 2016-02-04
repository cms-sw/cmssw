#ifndef Masks_H
#define Masks_H

/*!
  \file Masks.h
  \brief channel masking
  \author G. Della Ricca
  \version $Revision: 1.7 $
  \date $Date: 2010/08/20 19:09:03 $
*/

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

  static bool maskChannel( int ism, int i1, int i2, uint32_t bits, const EcalSubdetector subdet ) throw( std::runtime_error );

  static bool maskPn( int ism, int i1, uint32_t bits, const EcalSubdetector subdet ) throw( std::runtime_error );

private:

  Masks() {}; // Hidden to force static use
  ~Masks() {}; // Hidden to force static use

  static bool init;

  static const EcalDQMChannelStatus* channelStatus;
  static const EcalDQMTowerStatus* towerStatus;

};

#endif // Masks_H
