#ifndef EVENTFILTER_CASTORRAWTODIGI_CASTORPACKER_H
#define EVENTFILTER_CASTORRAWTODIGI_CASTORPACKER_H 1

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "CondFormats/CastorObjects/interface/CastorElectronicsMap.h"
#include "EventFilter/CastorRawToDigi/interface/CastorCollections.h"

/** \class CastorPacker
  *  
  * \author J. Mans - Minnesota
  */
class CastorPacker {
public:

  static void pack(int fedid, int dccnumber,
                   int nl1a, int orbitn, int bcn,
                   const CastorCollections& inputs, 
                   const CastorElectronicsMap& emap,
                   FEDRawData& output);
private:
  static int findSamples(const DetId& did, const CastorCollections& inputs,
                         unsigned short* buffer, int &presamples);
};

#endif
