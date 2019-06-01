#ifndef GEOMETRY_RECORDS_DD_SPECPAR_REGISTRY_RCD_H
#define GEOMETRY_RECORDS_DD_SPECPAR_REGISTRY_RCD_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "boost/mpl/vector.hpp"

class DDSpecParRegistryRcd
    : public edm::eventsetup::DependentRecordImplementation<DDSpecParRegistryRcd, boost::mpl::vector<GeometryFileRcd>> {
};

#endif
