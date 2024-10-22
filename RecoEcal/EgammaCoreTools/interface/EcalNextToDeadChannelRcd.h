#ifndef EcalNextToDeadChannelRcd_h
#define EcalNextToDeadChannelRcd_h

#include "FWCore/Utilities/interface/mplVector.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

//
// Registration of to the EventSetup mechanism
//

class EcalNextToDeadChannelRcd
    : public edm::eventsetup::DependentRecordImplementation<EcalNextToDeadChannelRcd,
                                                            edm::mpl::Vector<EcalChannelStatusRcd> > {};

#endif
