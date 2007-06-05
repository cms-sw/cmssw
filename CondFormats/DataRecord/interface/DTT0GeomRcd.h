#ifndef DTT0GEOMRCD_H
#define DTT0GEOMRCD_H

#include "boost/mpl/vector.hpp"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
class DTT0GeomRcd : public edm::eventsetup::DependentRecordImplementation<DTT0GeomRcd, boost::mpl::vector<MuonGeometryRecord> > {};
#endif
