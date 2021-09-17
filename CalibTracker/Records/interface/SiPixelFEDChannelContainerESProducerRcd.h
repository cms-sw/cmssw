#ifndef CALIBTRACKER_RECORDS_SIPIXELFEDCHANNELCONTAINERESPRODUCERRCD_H
#define CALIBTRACKER_RECORDS_SIPIXELFEDCHANNELCONTAINERESPRODUCERRCD_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"
#include "CondFormats/DataRecord/interface/SiPixelStatusScenariosRcd.h"

class SiPixelFEDChannelContainerESProducerRcd
    : public edm::eventsetup::DependentRecordImplementation<SiPixelFEDChannelContainerESProducerRcd,
                                                            edm::mpl::Vector<SiPixelStatusScenariosRcd> > {};

#endif
