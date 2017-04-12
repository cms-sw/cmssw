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

//move to the class
//members lower case 
struct localmaporder {
   std::map<int, std::map<int,int> > FEDmap {};
   std::map<int, std::map<int,int> > FEDorder {};

   localmaporder() = default;
};

class TowerBlockFormatter : public BlockFormatter {
 public :

	TowerBlockFormatter(EcalDigiToRaw *es) : BlockFormatter(es) {};
	~TowerBlockFormatter();
        static const int kCardsPerTower = 5;     // Number of VFE cards per trigger tower
        void DigiToRaw(const EBDataFrame& dataframe, FEDRawData& rawdata, const EcalElectronicsMapping* TheMapping, int bx, int lv1, localmaporder &local) const;
        void DigiToRaw(const EEDataFrame& dataframe, FEDRawData& rawdata, const EcalElectronicsMapping* TheMapping, int bx, int lv1, localmaporder &local) const;
	localmaporder StartEvent();
	void EndEvent(FEDRawDataCollection* productRawData);

 private :

};



#endif


