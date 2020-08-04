#ifndef EcalSeverityLevelAlgoRcd_h
#define EcalSeverityLevelAlgoRcd_h

#include "FWCore/Utilities/interface/mplVector.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

//
// Registration of EcalSeverityLevelAlgo to the EventSetup mechanism
//

class EcalSeverityLevelAlgoRcd
    : public edm::eventsetup::DependentRecordImplementation<EcalSeverityLevelAlgoRcd,
                                                            edm::mpl::Vector<EcalChannelStatusRcd> > {};

#endif
