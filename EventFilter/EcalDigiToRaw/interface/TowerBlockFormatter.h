#ifndef TOWERBLOCKFORMATTER_H
#define TOWERBLOCKFORMATTER_H

#include <iostream>
#include <vector>
#include <map>

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "EventFilter/EcalDigiToRaw/interface/BlockFormatter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"


// 
// The crystals corresponding to a given FE in a given FED
//

class TowerBlockFormatter : public BlockFormatter {
  public :
  struct FEDMapOrder {
    std::map<int, std::map<int,int> > fedmap {};
    std::map<int, std::map<int,int> > fedorder {};
    FEDMapOrder() = default;
  };

  TowerBlockFormatter(EcalDigiToRaw *es) : BlockFormatter(es) {};
  static const int kCardsPerTower = 5;     // Number of VFE cards per trigger tower
  void DigiToRaw(const EBDataFrame& dataframe, FEDRawData& rawdata, const EcalElectronicsMapping* TheMapping, int bx, int lv1, FEDMapOrder &local) const;
  void DigiToRaw(const EEDataFrame& dataframe, FEDRawData& rawdata, const EcalElectronicsMapping* TheMapping, int bx, int lv1, FEDMapOrder &local) const;
  inline FEDMapOrder StartEvent() const {return FEDMapOrder();}
  void EndEvent(FEDRawDataCollection* productRawData);

  private :

};



#endif


