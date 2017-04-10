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
        SRBlockFormatter(EcalDigiToRaw* es): BlockFormatter(es) {};
	~SRBlockFormatter();
        void DigiToRaw(int dccid, int dcc_channel, int flag, FEDRawData& rawdata, int bx, int lv1, std::map<int, int>& header_) const;

	std::map<int, int> StartEvent();

 private :
};



#endif


