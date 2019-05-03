#ifndef EcalSeverityLevelAlgoRcd_h
#define EcalSeverityLevelAlgoRcd_h

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "boost/mpl/vector.hpp"

//
// Registration of EcalSeverityLevelAlgo to the EventSetup mechanism
//

class EcalSeverityLevelAlgoRcd
    : public edm::eventsetup::DependentRecordImplementation<
          EcalSeverityLevelAlgoRcd, boost::mpl::vector<EcalChannelStatusRcd>> {
};

#endif
