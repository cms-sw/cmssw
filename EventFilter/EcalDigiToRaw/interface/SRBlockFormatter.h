#ifndef SRBLOCKFORMATTER_H
#define SRBLOCKFORMATTER_H

#include <iostream>
#include <vector>
#include <map>

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "EventFilter/EcalDigiToRaw/interface/BlockFormatter.h"




class SRBlockFormatter : public BlockFormatter {
 public :

	SRBlockFormatter();
	~SRBlockFormatter();
        void DigiToRaw(int dccid, int dcc_channel, int flag, FEDRawData& rawdata);

	void StartEvent();

 private :
	std::map<int, int> header_;
};



#endif


