#ifndef BLOCKFORMATTER_H
#define BLOCKFORMATTER_H

#include <iostream>
#include <vector>
#include <map>

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>



class BlockFormatter {
 public :
        typedef uint64_t Word64;
        typedef uint16_t Word16;
        
        struct Config {
          const std::vector<int32_t> * plistDCCId_;
          bool debug_;
          
          bool doBarrel_;
          bool doEndCap_;
          bool doTCC_;
          bool doSR_;
          bool doTower_;
        };
        struct Params {
          int counter_;
          int orbit_number_;
          int bx_;
          int lv1_;
          int runnumber_;
        };

	explicit BlockFormatter(Config const& iC, Params const& iP);
        static const int kCardsPerTower = 5;     // Number of VFE cards per trigger tower
	void DigiToRaw(FEDRawDataCollection* productRawData);
	void print(FEDRawData& rawdata);
	// void CleanUp(FEDRawDataCollection* productRawData);
	void CleanUp(FEDRawDataCollection* productRawData,
			std::map<int, std::map<int,int> >* FEDorder);
	void PrintSizes(FEDRawDataCollection* productRawData);

 protected :

        const std::vector<int32_t> * plistDCCId_;

        int counter_;
	int orbit_number_;
	int bx_;
	int lv1_;
	int  runnumber_;

        const bool debug_;

	const bool doBarrel_;
	const bool doEndCap_;
	const bool doTCC_;
        const bool doSR_;
        const bool doTower_;

};



#endif


