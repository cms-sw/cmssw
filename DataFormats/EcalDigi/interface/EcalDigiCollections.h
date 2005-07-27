#ifndef DIGIECAL_ECALDIGICOLLECTION_H
#define DIGIECAL_ECALDIGICOLLECTION_H

#include <vector>
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"

namespace cms {

typedef std::vector<EBDataFrame> EBDigiCollection;
typedef std::vector<EEDataFrame> EEDigiCollection;
typedef std::vector<EcalTriggerPrimitiveDigi> EcalTrigPrimDigiCollection;

}

#endif
