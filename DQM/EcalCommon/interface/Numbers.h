#ifndef NUMBERS_H
#define NUMBERS_H

/*!
  \file Numbers.h
  \brief Some "id" conversions
  \author B. Gobbo 
  \version $Revision: 1.38 $
  \date $Date: 2011/08/05 10:34:43 $
*/

#include <string>
#include <stdexcept>
#include <vector>

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

class DetId;
class EBDetId;
class EEDetId;

class EcalTrigTowerDetId;
class EcalElectronicsId;
class EcalPnDiodeDetId;
class EcalScDetId;

class EcalDCCHeaderBlock;

class EcalElectronicsMapping;
class EcalTrigTowerConstituentsMap;

class CaloGeometry;

class Numbers {

 public:

  static void initGeometry( const edm::EventSetup& setup, bool verbose = false );

  static int iEB( const int ism ) throw( std::runtime_error );

  static std::string sEB( const int ism );

  static int iEE( const int ism ) throw( std::runtime_error );

  static std::string sEE( const int ism );

  static EcalSubdetector subDet( const EBDetId& id );

  static EcalSubdetector subDet( const EEDetId& id );

  static EcalSubdetector subDet( const EcalTrigTowerDetId& id );

  static EcalSubdetector subDet( const EcalScDetId& id );

  static EcalSubdetector subDet( const EcalElectronicsId& id );

  static EcalSubdetector subDet( const EcalPnDiodeDetId& id );

  static EcalSubdetector subDet( const EcalDCCHeaderBlock& id ) throw( std::runtime_error );

  static int iSM( const int ism, const EcalSubdetector subdet ) throw( std::runtime_error );

  static int iSM( const EBDetId& id ) throw( std::runtime_error );

  static int iSM( const EEDetId& id ) throw( std::runtime_error );

  static int iSM( const EcalTrigTowerDetId& id ) throw( std::runtime_error );

  static int iSM( const EcalElectronicsId& id ) throw( std::runtime_error );

  static int iSM( const EcalPnDiodeDetId& id ) throw( std::runtime_error );

  static int iSM( const EcalScDetId& id ) throw( std::runtime_error );

  static int iSM( const EcalDCCHeaderBlock& id, const EcalSubdetector subdet ) throw( std::runtime_error );

  static int iSC( const EcalScDetId& id ) throw( std::runtime_error );

  static int iSC( const int ism, const EcalSubdetector subdet, const int i1, const int i2 ) throw( std::runtime_error );

  static int iTT( const int ism, const EcalSubdetector subdet, const int i1, const int i2 ) throw( std::runtime_error );

  static int iTT( const EcalTrigTowerDetId& id ) throw( std::runtime_error );

  static int iTCC(const int ism, const EcalSubdetector subdet, const int i1, const int i2) throw( std::runtime_error );

  static int iTCC(const EcalTrigTowerDetId& id) throw( std::runtime_error );

  static std::vector<DetId>* crystals( const EcalTrigTowerDetId& id ) throw( std::runtime_error );

  static std::vector<DetId>* crystals( const EcalElectronicsId& id ) throw( std::runtime_error );

  static std::vector<DetId>* crystals( int idcc, int isc ) throw( std::runtime_error );

  static const EcalScDetId getEcalScDetId( const EEDetId& id ) throw( std::runtime_error );

  static int indexEB( const int ism, const int ie, const int ip );

  static int indexEE( const int ism, const int ix, const int iy );

  static int icEB( const int ism, const int ix, const int iy );

  static int icEE( const int ism, const int ix, const int iy ) throw( std::runtime_error );

  static int RtHalf(const EBDetId& id);

  static int RtHalf(const EEDetId& id);

  static int ix0EE( const int ism );

  // returns ix0 in negative-number scheme for EE- instead of 101-ix
  static int ix0EEm( const int ism );

  static int iy0EE( const int ism );

  static bool validEE( const int ism, const int ix, const int iy );

  static bool validEESc( const int ism, const int ix, const int iy );

  static const EcalElectronicsMapping* getElectronicsMapping() throw( std::runtime_error );

  // temporary - this is not really an "id conversion" - must find a better place to implement
  static float eta( const DetId &id );
  static float phi( const DetId &id );

private:

  Numbers() {}; // Hidden to force static use
  ~Numbers() {}; // Hidden to force static use

  static bool init;

  static const EcalElectronicsMapping* map;
  static const EcalTrigTowerConstituentsMap* mapTT;

  static const CaloGeometry *geometry;

  static std::vector<DetId> crystalsTCC_[100*108];
  static std::vector<DetId> crystalsDCC_[100* 54];

};

#endif // NUMBERS_H
