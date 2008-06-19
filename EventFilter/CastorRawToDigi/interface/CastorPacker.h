#ifndef EVENTFILTER_CASTORRAWTODIGI_CASTORPACKER_H
#define EVENTFILTER_CASTORRAWTODIGI_CASTORPACKER_H 1

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "CondFormats/CastorObjects/interface/CastorElectronicsMap.h"

/** \class HcalPacker
  *  
  * $Date: 2007/02/19 04:05:40 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class CastorPacker {
public:
  struct Collections {
    Collections();
    const CastorDigiCollection* castorCont;
    const HcalCalibDigiCollection* calibCont;
    const HcalTrigPrimDigiCollection* tpCont;
  };

  void pack(int fedid, int dccnumber,
	    int nl1a, int orbitn, int bcn,
	    const Collections& inputs, 
	    const CastorElectronicsMap& emap,
	    FEDRawData& output);
private:
  int findSamples(const DetId& did, const Collections& inputs,
		  unsigned short* buffer, int &presamples);
};

#endif
