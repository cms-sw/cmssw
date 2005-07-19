#ifndef DIGIHCAL_HCALDIGICOLLECTION_H
#define DIGIHCAL_HCALDIGICOLLECTION_H

#include <vector>
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"

typedef std::vector<HBHEDataFrame> HBHEDigiCollection;
typedef std::vector<HODataFrame> HODigiCollection;
typedef std::vector<HFDataFrame> HFDigiCollection;
typedef std::vector<HcalTriggerPrimitiveDigi> HcalTrigPrimDigiCollection;

#endif
