#ifndef BLOCKFORMATTER_H
#define BLOCKFORMATTER_H

#include <iostream>
#include <vector>
#include <map>

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>



 class EcalDigiToRaw;


class BlockFormatter {
 public :
        typedef uint64_t Word64;
        typedef uint16_t Word16;

	BlockFormatter(EcalDigiToRaw* base);
	~BlockFormatter();
        static const int kCardsPerTower = 5;     // Number of VFE cards per trigger tower
	void DigiToRaw(FEDRawDataCollection* productRawData, int run_number, int orbit_number, int bx, int lv1) const;
	void print(FEDRawData& rawdata) const;
	void CleanUp(FEDRawDataCollection& productRawData,
			std::map<int, std::map<int,int> >& FEDorder);
	void PrintSizes(FEDRawDataCollection* productRawData) const;

 protected :

        const bool debug_;

	const bool doBarrel_;
	const bool doEndCap_;
        const std::vector<int32_t> * plistDCCId_;
	const bool doTCC_;
        const bool doSR_;
        const bool doTower_;

};



#endif


