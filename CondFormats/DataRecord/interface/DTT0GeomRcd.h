#ifndef DTT0GEOMRCD_H
#define DTT0GEOMRCD_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h.h"
class DTT0Rcd : public edm::eventsetup::DependentRecordImplementation<DTT0GeomRcd, boost::mpl::vector<MuonGeometryRecord> > {};
#endif
