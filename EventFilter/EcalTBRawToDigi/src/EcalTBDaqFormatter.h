#ifndef EcalTBDaqFormatter_H
#define EcalTBDaqFormatter_H
/** \class EcalTBDaqFormatter
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Marinelli  IASA-Athens
 *
 */
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <vector> 
#include <map>
using namespace std;
#include <iostream>

namespace raw {class FEDRawData;}
class DCCDataParser;
class EcalTBDaqFormatter   {

 public:


  EcalTBDaqFormatter();
  virtual ~EcalTBDaqFormatter(){cout << " Destroying EcalTBDaqFormatter " << endl; };
  
  
  void  interpretRawData( const raw::FEDRawData & data , cms::EBDigiCollection& digicollection );
  
 private:
  
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

  
  
};
#endif
