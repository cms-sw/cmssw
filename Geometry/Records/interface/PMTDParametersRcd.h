#ifndef PMTDParametersRcd_H
#define PMTDParametersRcd_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

class PMTDParametersRcd
    : public edm::eventsetup::DependentRecordImplementation<PMTDParametersRcd, edm::mpl::Vector<IdealGeometryRecord> > {
};

#endif  // PMTDParameters_H
