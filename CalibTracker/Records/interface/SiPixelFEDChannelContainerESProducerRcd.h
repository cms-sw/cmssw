#ifndef CALIBTRACKER_RECORDS_SIPIXELFEDCHANNELCONTAINERESPRODUCERRCD_H
#define CALIBTRACKER_RECORDS_SIPIXELFEDCHANNELCONTAINERESPRODUCERRCD_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "boost/mpl/vector.hpp"
#include "CondFormats/DataRecord/interface/SiPixelStatusScenariosRcd.h"

class SiPixelFEDChannelContainerESProducerRcd
    : public edm::eventsetup::DependentRecordImplementation<SiPixelFEDChannelContainerESProducerRcd,
                                                            boost::mpl::vector<SiPixelStatusScenariosRcd> > {};

#endif
