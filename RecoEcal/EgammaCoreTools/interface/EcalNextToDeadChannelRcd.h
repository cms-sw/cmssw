#ifndef EcalNextToDeadChannelRcd_h
#define EcalNextToDeadChannelRcd_h

#include <boost/mp11/list.hpp>
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

//
// Registration of to the EventSetup mechanism
//

class EcalNextToDeadChannelRcd
    : public edm::eventsetup::DependentRecordImplementation<EcalNextToDeadChannelRcd,
                                                            boost::mp11::mp_list<EcalChannelStatusRcd> > {};

#endif
