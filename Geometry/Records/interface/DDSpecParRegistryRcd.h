#ifndef GEOMETRY_RECORDS_DD_SPECPAR_REGISTRY_RCD_H
#define GEOMETRY_RECORDS_DD_SPECPAR_REGISTRY_RCD_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

class DDSpecParRegistryRcd
    : public edm::eventsetup::DependentRecordImplementation<DDSpecParRegistryRcd, edm::mpl::Vector<IdealGeometryRecord>> {
};

#endif
