#ifndef TOWERBLOCKFORMATTER_H
#define TOWERBLOCKFORMATTER_H

#include <iostream>
#include <vector>
#include <map>

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "EventFilter/EcalDigiToRaw/interface/BlockFormatter.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"


// 
// The crystals corresponding to a given FE in a given FED
//



class TowerBlockFormatter : public BlockFormatter {
 public :
  
  TowerBlockFormatter(BlockFormatter::Config const&, BlockFormatter::Params const& );

  static const int kCardsPerTower = 5;     // Number of VFE cards per trigger tower
  void DigiToRaw(const EBDataFrame& dataframe, FEDRawData& rawdata, const EcalElectronicsMapping* TheMapping);
  void DigiToRaw(const EEDataFrame& dataframe, FEDRawData& rawdata, const EcalElectronicsMapping* TheMapping);
  void EndEvent(FEDRawDataCollection* productRawData);
  
  std::map<int, std::map<int,int> >& GetFEDorder() {return FEDorder; }
  
 private :
  std::map<int, std::map<int,int> > FEDmap;
  std::map<int, std::map<int,int> > FEDorder;


};



#endif


