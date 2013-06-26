#ifndef NUMBERS_H
#define NUMBERS_H

/*!
  \file Numbers.h
  \brief Some "id" conversions
  \author B. Gobbo 
  \version $Revision: 1.43 $
  \date $Date: 2012/04/27 13:46:03 $
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

  static int iEB( const unsigned ism );

  static std::string sEB( const unsigned ism );

  static int iEE( const unsigned ism );

  static std::string sEE( const unsigned ism );

  static EcalSubdetector subDet( const EBDetId& id );

  static EcalSubdetector subDet( const EEDetId& id );

  static EcalSubdetector subDet( const EcalTrigTowerDetId& id );

  static EcalSubdetector subDet( const EcalScDetId& id );

  static EcalSubdetector subDet( const EcalElectronicsId& id );

  static EcalSubdetector subDet( const EcalPnDiodeDetId& id );

  static EcalSubdetector subDet( const EcalDCCHeaderBlock& id );

  // for EB, converts between two schemes. Old scheme [1:9] for EB-, new scheme (used in EBDetId) [1:9] for EB+
  static unsigned iSM( const unsigned ism, const EcalSubdetector subdet );

  static unsigned iSM( const EBDetId& id );

  static unsigned iSM( const EEDetId& id );

  static unsigned iSM( const EcalTrigTowerDetId& id );

  static unsigned iSM( const EcalElectronicsId& id );

  static unsigned iSM( const EcalPnDiodeDetId& id );

  static unsigned iSM( const EcalScDetId& id );

  static unsigned iSM( const EcalDCCHeaderBlock& id, const EcalSubdetector subdet );

  static unsigned iSC( const EcalScDetId& id );

  static unsigned iSC( const unsigned ism, const EcalSubdetector subdet, const unsigned i1, const unsigned i2 );

  static unsigned iTT( const unsigned ism, const EcalSubdetector subdet, const unsigned i1, const unsigned i2 );

  static unsigned iTT( const EcalTrigTowerDetId& id );

  static unsigned iTCC(const unsigned ism, const EcalSubdetector subdet, const unsigned i1, const unsigned i2);

  static unsigned iTCC(const EcalTrigTowerDetId& id);

  static std::vector<DetId>* crystals( const EcalTrigTowerDetId& id );

  static std::vector<DetId>* crystals( const EcalElectronicsId& id );

  static std::vector<DetId>* crystals( unsigned idcc, unsigned isc );

  static const EcalScDetId getEcalScDetId( const EEDetId& id );

  static unsigned indexEB( const unsigned ism, const unsigned ie, const unsigned ip );

  static unsigned indexEE( const unsigned ism, const unsigned ix, const unsigned iy );

  static unsigned icEB( const unsigned ism, const unsigned ix, const unsigned iy );

  static unsigned icEE( const unsigned ism, const unsigned ix, const unsigned iy );

  static unsigned RtHalf(const EBDetId& id);

  static unsigned RtHalf(const EEDetId& id);

  static int ix0EE( const unsigned ism );

  // returns ix0 in negative-number scheme for EE- instead of 101-ix
  static int ix0EEm( const unsigned ism );

  static int iy0EE( const unsigned ism );

  static bool validEE( const unsigned ism, const unsigned ix, const unsigned iy );

  static bool validEESc( const unsigned ism, const unsigned ix, const unsigned iy );

  static unsigned nCCUs(const unsigned ism);

  static unsigned nTTs(const unsigned itcc);

  static const EcalElectronicsMapping* getElectronicsMapping();

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

  static const unsigned crystalsTCCArraySize_ = 100 * 108;
  static const unsigned crystalsDCCArraySize_ = 100 * 54;

  static std::vector<DetId> crystalsTCC_[crystalsTCCArraySize_];
  static std::vector<DetId> crystalsDCC_[crystalsDCCArraySize_];

};

#endif // NUMBERS_H
