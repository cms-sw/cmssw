#ifndef EcalNextToDeadChannelRcd_h
#define EcalNextToDeadChannelRcd_h


#include "boost/mpl/vector.hpp"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

//
// Registration of to the EventSetup mechanism
//

class EcalNextToDeadChannelRcd : public edm::eventsetup::DependentRecordImplementation<EcalNextToDeadChannelRcd, boost::mpl::vector<EcalChannelStatusRcd> > {};

#endif
