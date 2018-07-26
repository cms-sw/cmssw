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

  SRBlockFormatter(BlockFormatter::Config const&, BlockFormatter::Params const& );

  void DigiToRaw(int dccid, int dcc_channel, int flag, FEDRawData& rawdata);
  
 private :
  std::map<int, int> header_;
};

#endif


