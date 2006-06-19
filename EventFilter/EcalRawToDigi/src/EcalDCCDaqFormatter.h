#ifndef EcalDCCDaqFormatter_H
#define EcalDCCDaqFormatter_H
/** \class EcalDCCDaqFormatter
 *
 *  $Date: 2006/04/27 21:56:57 $
 *  $Revision: 1.13 $
 *  \author N. Marinelli  IASA-Athens
 *  \author G. Della Ricca
 *  \author G. Franzoni
 *  \author A. Ghezzi
 *  \author P. Meridiani
 *
 *  TODO
 *  This version is not yet suitable to be used for EcalEndcaps
 *  Waiting for the final ElectronicsMapping 
 */

#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>
#include "EventFilter/EcalRawToDigi/src/DCCTowerBlock.h"
#include "EventFilter/EcalRawToDigi/src/DCCDataParser.h"

#include <vector> 
#include <map>
#include <iostream>

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

class FEDRawData;
class DCCDataParser;
class DCCMapper;
class EcalDCCDaqFormatter   {

 public:

  EcalDCCDaqFormatter();

  virtual ~EcalDCCDaqFormatter()
    {
      LogDebug("EcalRawToDigi") << "@SUB=EcalDCCDaqFormatter" << "\n"; 
      delete theParser_;
    };


  void setDCCMapper(const DCCMapper* mapper) { theMapper_=mapper; } ; 
  void setEcalFirstFED(const int& ecalFirstFED) { ecalFirstFED_=ecalFirstFED; } ; 
  
  void  interpretRawData( const FEDRawData & data , EBDigiCollection& digicollection , EcalPnDiodeDigiCollection & pndigicollection ,
			  EcalRawDataCollection& DCCheaderCollection,
			  EBDetIdCollection & dccsizecollection ,
			  EcalTrigTowerDetIdCollection & ttidcollection , EcalTrigTowerDetIdCollection & blocksizecollection,
			  EBDetIdCollection & chidcollection , EBDetIdCollection & gaincollection ,
			  EBDetIdCollection & gainswitchcollection , EBDetIdCollection & gainswitchstaycollection,
			  EcalElectronicsIdCollection & memttidcollection,  EcalElectronicsIdCollection &  memblocksizecollection,
			  EcalElectronicsIdCollection & memgaincollection,  EcalElectronicsIdCollection & memchidcollection);
 

 private:
  
  void  DecodeMEM( int DCCid, DCCTowerBlock *  towerblock, EcalPnDiodeDigiCollection & pndigicollection ,
		   EcalElectronicsIdCollection & memttidcollection,  EcalElectronicsIdCollection &  memblocksizecollection,
		   EcalElectronicsIdCollection & memgaincollection,  EcalElectronicsIdCollection & memchidcollection);
  
  // Mantain this here as long as everything is moved to a general mapping
  pair<int,int>  cellIndex(int tower_id, int strip, int xtal); 
  int            cryIc(int tower_id, int strip, int xtal); 
  bool leftTower(int tower) const ;
  bool rightTower(int tower) const ;

 private:

  DCCDataParser* theParser_;

  int ecalFirstFED_;

  //Mapper between FedId and SMid (temporary solution for the moment)
  const DCCMapper* theMapper_;

  // Mantain this here as long as everything is moved to a general mapping
  enum SMGeom_t 
    {
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

  enum SMElectronics_t 
    {
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
