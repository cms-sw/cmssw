#ifndef DTT0RCD_H
#define DTT0RCD_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <vector>
class DTT0Rcd : public edm::eventsetup::DependentRecordImplementation<DTT0Rcd,
		boost::mpl::vector<IdealGeometryRecord, MuonNumberingRecord> > {};

//class DTT0Rcd : public edm::eventsetup::EventSetupRecordImplementation<DTT0Rcd> {};
#endif
