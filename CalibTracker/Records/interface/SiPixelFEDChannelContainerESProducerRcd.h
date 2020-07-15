#ifndef CALIBTRACKER_RECORDS_SIPIXELFEDCHANNELCONTAINERESPRODUCERRCD_H
#define CALIBTRACKER_RECORDS_SIPIXELFEDCHANNELCONTAINERESPRODUCERRCD_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include <boost/mp11/list.hpp>
#include "CondFormats/DataRecord/interface/SiPixelStatusScenariosRcd.h"

class SiPixelFEDChannelContainerESProducerRcd
    : public edm::eventsetup::DependentRecordImplementation<SiPixelFEDChannelContainerESProducerRcd,
                                                            boost::mp11::mp_list<SiPixelStatusScenariosRcd> > {};

#endif
