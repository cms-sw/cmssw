#ifndef EcalRegionCablingRecord_H
#define EcalRegionCablingRecord_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

class EcalRegionCablingRecord
    : public edm::eventsetup::DependentRecordImplementation<EcalRegionCablingRecord, edm::mpl::Vector<EcalMappingRcd> > {
};

#endif
