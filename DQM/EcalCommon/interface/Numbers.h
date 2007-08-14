// $Id: Numbers.h,v 1.7 2007/05/24 13:04:42 benigno Exp $

/*!
  \file Numbers.h
  \brief Some "id" conversions
  \author B. Gobbo 
  \version $Revision: 1.7 $
  \date $Date: 2007/05/24 13:04:42 $
*/

#ifndef Numbers_H
#define Numbers_H

#include <string>
#include <stdexcept>

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include <DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h>
#include <DataFormats/EcalDetId/interface/EcalElectronicsId.h>
#include <DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h>
#include <DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h>

class Numbers {

 public:

  static void        initGeometry( const edm::EventSetup& setup );

  static int         iEB( const int ism ) throw( std::runtime_error );

  static std::string sEB( const int ism ) throw( std::runtime_error );

  static int         iEE( const int ism ) throw( std::runtime_error );

  static std::string sEE( const int ism ) throw( std::runtime_error );

  static int         iSM( const int ism, const int subdet ) throw( std::runtime_error );

  static int         iSM( const EBDetId& id ) throw( std::runtime_error );

  static int         iSM( const EEDetId& id ) throw( std::runtime_error );

  static int         iSM( const EcalTrigTowerDetId& id );

  static int         iSM( const EcalElectronicsId&  id );

  static int         iSM( const EcalPnDiodeDetId&   id );

  static int         iSM( const EcalDCCHeaderBlock& id );

  static int ix0EE( const int ism );

  static int iy0EE( const int ism );

  static bool validEE( const int ism, const int ix, const int iy );

  static int icEE( const int ism, const int ix, const int iy );

  static int ixSectorsEE[202];
  static int iySectorsEE[202];

private:

  static bool init;

  static const EcalElectronicsMapping* map;

};

#endif // Numbers_H
