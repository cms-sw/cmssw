#ifndef Masks_H
#define Masks_H

/*!
  \file Masks.h
  \brief channel masking
  \author G. Della Ricca
  \version $Revision: 1.10 $
  \date $Date: 2012/04/27 13:46:03 $
*/

#include <string>
#include <stdexcept>
#include <vector>

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "CommonTools/Utils/interface/Exception.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

class Masks {

 public:

  static void initMasking( const edm::EventSetup& setup, bool verbose = false );

  static bool maskChannel( int ism, int i1, int i2, uint32_t bits, const EcalSubdetector subdet ) throw( cms::Exception );

  static bool maskPn( int ism, int i1, uint32_t bits, const EcalSubdetector subdet ) throw( cms::Exception );

private:

  Masks() {}; // Hidden to force static use
  ~Masks() {}; // Hidden to force static use

  static bool init;

  static const EcalDQMChannelStatus* channelStatus;
  static const EcalDQMTowerStatus* towerStatus;

};

#endif // Masks_H
