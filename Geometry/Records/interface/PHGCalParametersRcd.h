#ifndef Geometry_Records_PHGCalParametersRcd_H
#define Geometry_Records_PHGCalParametersRcd_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

class PHGCalParametersRcd
    : public edm::eventsetup::DependentRecordImplementation<PHGCalParametersRcd, edm::mpl::Vector<IdealGeometryRecord> > {
};

#endif  // PHGCalParameters_H
