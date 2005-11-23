#ifndef EcalTBDaqFormatter_H
#define EcalTBDaqFormatter_H
/** \class EcalTBDaqFormatter
 *
 *  $Date: 2005/10/18 09:06:15 $
 *  $Revision: 1.4 $
 *  \author N. Marinelli  IASA-Athens
 *  \author G. Della Ricca
 *  \author G. Franzoni
 *  \author A. Ghezzi
 *
 */
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <vector> 
#include <map>
using namespace std;
#include <iostream>

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class FEDRawData;
class DCCDataParser;
class EcalTBDaqFormatter   {

 public:


  EcalTBDaqFormatter(DaqMonitorBEInterface* dbe);
  virtual ~EcalTBDaqFormatter(){cout << " Destroying EcalTBDaqFormatter " << endl; };
  
  
  //  void  interpretRawData( const FEDRawData & data , EBDigiCollection& digicollection );
  void  interpretRawData( const FEDRawData & data , EBDigiCollection& digicollection , EcalPnDiodeDigiCollection & pndigicollection );
  
 private:
  
  void DecodeMEM(int tower_id);
  pair<int,int>  cellIndex(int tower_id, int strip, int xtal); 
  bool leftTower(int tower) const ;
  bool rightTower(int tower) const ;
  
 private:
  DCCDataParser* theParser_;
  
   
  enum SMGeom_t {
     kModules = 4,           // Number of modules per supermodule
     kTriggerTowers = 68,    // Number of trigger towers per supermodule
     kTowersInPhi = 4,       // Number of trigger towers in phi
     kTowersInEta = 17,      // Number of trigger towers in eta
     kCrystals = 1700,       // Number of crystals per supermodule
     kCrystalsInPhi = 20,    // Number of crystals in phi
     kCrystalsInEta = 85,    // Number of crystals in eta
     kCrystalsPerTower = 25, // Number of crystals per trigger tower
     kCardsPerTower = 5,     // Number of VFE cards per trigger tower
     kChannelsPerCard = 5    // Number of channels per VFE card
   };

  MonitorElement* meIntegrityChId[36];  
  MonitorElement* meIntegrityGain[36];
  MonitorElement* meIntegrityTTId[36];
  MonitorElement* meIntegrityTTBlockSize[36];
  MonitorElement* meIntegrityDCCSize;

  // used for mem boxes unpacking
  int fem[5][5][11];            // store raw data for one mem
  int data_MEM[500];      // collects unpacked data for both mems 

  
};
#endif
