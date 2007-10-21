// $Id: Numbers.h,v 1.14 2007/10/17 15:58:43 dellaric Exp $

/*!
  \file Numbers.h
  \brief Some "id" conversions
  \author B. Gobbo 
  \version $Revision: 1.14 $
  \date $Date: 2007/10/17 15:58:43 $
*/

#ifndef Numbers_H
#define Numbers_H

#include <string>
#include <stdexcept>
#include <vector>

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>

#include <DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h>
#include <DataFormats/EcalDetId/interface/EcalElectronicsId.h>
#include <DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h>
#include <DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h>

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

class Numbers {

 public:

  static void initGeometry( const edm::EventSetup& setup );

  static int iEB( const int ism ) throw( std::runtime_error );

  static std::string sEB( const int ism ) throw( std::runtime_error );

  static int iEE( const int ism ) throw( std::runtime_error );

  static std::string sEE( const int ism ) throw( std::runtime_error );

  static int iSM( const int ism, const int subdet ) throw( std::runtime_error );

  static int iSM( const EBDetId& id ) throw( std::runtime_error );

  static int iSM( const EEDetId& id ) throw( std::runtime_error );

  static int iSM( const EcalTrigTowerDetId& id ) throw( std::runtime_error );

  static int iSM( const EcalElectronicsId& id );

  static int iSM( const EcalPnDiodeDetId& id );

  static int iSM( const EcalDCCHeaderBlock& id, const int subdet );

  static int iTT( const int ism, const int subdet, const int i1, const int i2 ) throw( std::runtime_error );

  static int iTT( const EcalTrigTowerDetId& id ) throw( std::runtime_error );

  static int indexEB( const int ism, const int ie, const int ip );

  static int indexEE( const int ism, const int ix, const int iy );

  static int icEB( const int ism, const int ix, const int iy );

  static int icEE( const int ism, const int ix, const int iy );

  static std::vector<DetId> crystals( const EcalTrigTowerDetId& id ) throw( std::runtime_error );

  static std::vector<DetId> crystals( const EcalElectronicsId& id ) throw( std::runtime_error );

  static int ix0EE( const int ism );

  static int iy0EE( const int ism );

  static bool validEE( const int ism, const int ix, const int iy );

  static int ixSectorsEE[202];
  static int iySectorsEE[202];

  static int inTowersEE[400];

private:

  static bool init;

  static const EcalElectronicsMapping* map;

};

#endif // Numbers_H
