// $Id: Numbers.h,v 1.21 2008/01/05 09:18:35 dellaric Exp $

/*!
  \file Numbers.h
  \brief Some "id" conversions
  \author B. Gobbo 
  \version $Revision: 1.21 $
  \date $Date: 2008/01/05 09:18:35 $
*/

#ifndef Numbers_H
#define Numbers_H

#include <string>
#include <stdexcept>
#include <vector>

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

class EBDetId;
class EEDetId;

class EcalTrigTowerDetId;
class EcalElectronicsId;
class EcalPnDiodeDetId;

class EcalDCCHeaderBlock;

class EcalElectronicsMapping;

class Numbers {

 public:

  static void initGeometry( const edm::EventSetup& setup );

  static int iEB( const int ism ) throw( std::runtime_error );

  static std::string sEB( const int ism );

  static int iEE( const int ism ) throw( std::runtime_error );

  static std::string sEE( const int ism );

  static EcalSubdetector subDet( const EBDetId& id );

  static EcalSubdetector subDet( const EEDetId& id );

  static EcalSubdetector subDet( const EcalTrigTowerDetId& id );

  static EcalSubdetector subDet( const EcalElectronicsId& id );

  static EcalSubdetector subDet( const EcalPnDiodeDetId& id );

  static EcalSubdetector subDet( const EcalDCCHeaderBlock& id ) throw( std::runtime_error );

  static int iSM( const int ism, const EcalSubdetector subdet ) throw( std::runtime_error );

  static int iSM( const EBDetId& id ) throw( std::runtime_error );

  static int iSM( const EEDetId& id ) throw( std::runtime_error );

  static int iSM( const EcalTrigTowerDetId& id ) throw( std::runtime_error );

  static int iSM( const EcalElectronicsId& id ) throw( std::runtime_error );

  static int iSM( const EcalPnDiodeDetId& id ) throw( std::runtime_error );

  static int iSM( const EcalDCCHeaderBlock& id, const EcalSubdetector subdet ) throw( std::runtime_error );

  static int iTT( const int ism, const EcalSubdetector subdet, const int i1, const int i2 ) throw( std::runtime_error );

  static int iTT( const EcalTrigTowerDetId& id ) throw( std::runtime_error );

  static int indexEB( const int ism, const int ie, const int ip );

  static int indexEE( const int ism, const int ix, const int iy );

  static int icEB( const int ism, const int ix, const int iy );

  static int icEE( const int ism, const int ix, const int iy ) throw( std::runtime_error );

  static std::vector<DetId> crystals( const EcalTrigTowerDetId& id ) throw( std::runtime_error );

  static std::vector<DetId> crystals( const EcalElectronicsId& id ) throw( std::runtime_error );

  static int RtHalf(const EBDetId& id);

  static int RtHalf(const EEDetId& id);

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
