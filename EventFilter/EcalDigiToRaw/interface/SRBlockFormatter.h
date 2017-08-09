#ifndef SRBLOCKFORMATTER_H
#define SRBLOCKFORMATTER_H

#include <iostream>
#include <vector>
#include <map>

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "EventFilter/EcalDigiToRaw/interface/BlockFormatter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



class SRBlockFormatter : public BlockFormatter {
  public :
    SRBlockFormatter(EcalDigiToRaw* es): BlockFormatter(es) {};
    void DigiToRaw(int dccid, int dcc_channel, int flag, FEDRawData& rawdata, int bx, int lv1, std::map<int, int>& header_) const;

    inline std::map<int, int> StartEvent() const { 
      std::map<int, int> header;
      header.clear();
      return header;
    }
  private :
};



#endif


