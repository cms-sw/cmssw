#ifndef EcalTBDaqFormatter_H
#define EcalTBDaqFormatter_H
/** \class EcalTBDaqFormatter
 *
 *  $Date: 2007/10/23 15:30:05 $
 *  $Revision: 1.17 $
 *  \author N. Marinelli  IASA-Athens
 *  \author G. Della Ricca
 *  \author G. Franzoni
 *  \author A. Ghezzi
 *
 */
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>
#include "DCCTowerBlock.h"

#include <vector> 
#include <map>
#include <iostream>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


class FEDRawData;
class DCCTBDataParser;
class EcalTBDaqFormatter   {

 public:

  EcalTBDaqFormatter();
  virtual ~EcalTBDaqFormatter(){LogDebug("EcalTBRawToDigi") << "@SUB=EcalTBDaqFormatter" << "\n"; };

  void  interpretRawData( const FEDRawData & data , EBDigiCollection& digicollection , EcalPnDiodeDigiCollection & pndigicollection ,
			  EcalRawDataCollection& DCCheaderCollection,
			  EBDetIdCollection & dccsizecollection,
			  EcalElectronicsIdCollection & ttidcollection , EcalElectronicsIdCollection & blocksizecollection,
			  EBDetIdCollection & chidcollection , EBDetIdCollection & gaincollection ,
			  EBDetIdCollection & gainswitchcollection , 
			  EcalElectronicsIdCollection & memttidcollection,  EcalElectronicsIdCollection &  memblocksizecollection,
			  EcalElectronicsIdCollection & memgaincollection,  EcalElectronicsIdCollection & memchidcollection,
			  EcalTrigPrimDigiCollection &tpcollection);
 

 private:
  
  void  DecodeMEM( DCCTBTowerBlock *  towerblock, EcalPnDiodeDigiCollection & pndigicollection ,
		   EcalElectronicsIdCollection & memttidcollection,  EcalElectronicsIdCollection &  memblocksizecollection,
		   EcalElectronicsIdCollection & memgaincollection,  EcalElectronicsIdCollection & memchidcollection);
  
  std::pair<int,int>  cellIndex(int tower_id, int strip, int xtal); 
  int            cryIc(int tower_id, int strip, int xtal); 
  bool leftTower(int tower) const ;
  bool rightTower(int tower) const ;

 private:
  DCCTBDataParser* theParser_;

  enum SMGeom_t {
     kModules = 4,           // Number of modules per supermodule
     kTriggerTowers = 68,    // Number of trigger towers per supermodule
     kTowersInPhi = 4,       // Number of trigger towers in phi
     kTowersInEta = 17,      // Number of trigger towers in eta
     kCrystals = 1700,       // Number of crystals per supermodule
     kPns = 10,                  // Number of PN laser monitoring diodes per supermodule
     kCrystalsInPhi = 20,    // Number of crystals in phi
     kCrystalsInEta = 85,    // Number of crystals in eta
     kCrystalsPerTower = 25, // Number of crystals per trigger tower
     kCardsPerTower = 5,     // Number of VFE cards per trigger tower
     kChannelsPerCard = 5    // Number of channels per VFE card
   };

  enum SMElectronics_t {
    kSamplesPerChannel = 10,  // Number of sample per channel, per event
    kSamplesPerPn          = 50,  // Number of sample per PN, per event
    kChannelsPerTower   = 25,  // Number of channels per trigger tower
    kStripsPerTower        = 5,   // Number of VFE cards per trigger tower
    kChannelsPerStrip     = 5,   // Number channels per VFE card
    kPnPerTowerBlock    = 5,    // Number Pn diodes pertaining to 1 tower block = 1/2 mem box
    kTriggerTowersAndMem  = 70    // Number of trigger towers block including mems
  };

  // index and container for expected towers (according to DCC status)
  unsigned _numExpectedTowers;
  unsigned _ExpectedTowers[71];
  unsigned _expTowersIndex;

  // used for mem boxes unpacking
  int    memRawSample_[kStripsPerTower][kChannelsPerStrip][ kSamplesPerChannel+1];       // store raw data for one mem
  int    data_MEM[500];                                                                                                                  // collects unpacked data for both mems 
  bool pnAllocated;
  bool pnIsOkInBlock[kPnPerTowerBlock];

};
#endif
