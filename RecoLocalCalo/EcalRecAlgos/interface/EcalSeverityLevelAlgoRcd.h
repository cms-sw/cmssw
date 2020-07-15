#ifndef EcalSeverityLevelAlgoRcd_h
#define EcalSeverityLevelAlgoRcd_h

#include <boost/mp11/list.hpp>
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

//
// Registration of EcalSeverityLevelAlgo to the EventSetup mechanism
//

class EcalSeverityLevelAlgoRcd
    : public edm::eventsetup::DependentRecordImplementation<EcalSeverityLevelAlgoRcd,
                                                            boost::mp11::mp_list<EcalChannelStatusRcd> > {};

#endif
