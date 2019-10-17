#ifndef GEOMETRY_RECORDS_DD_VECTOR_REGISTRY_RCD_H
#define GEOMETRY_RECORDS_DD_VECTOR_REGISTRY_RCD_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "boost/mpl/vector.hpp"

class DDVectorRegistryRcd
    : public edm::eventsetup::DependentRecordImplementation<DDVectorRegistryRcd,
                                                            boost::mpl::vector<IdealGeometryRecord>> {};

#endif
