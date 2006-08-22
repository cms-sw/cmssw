#ifndef Records_MuonGeometryRecord_h
#define Records_MuonGeometryRecord_h

/** \class MuonGeometryRecord
 *  The Muon DetUnit geometry.
 *
 *  $Date: 2005/10/25 14:10:07 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "boost/mpl/vector.hpp"
#include "CondFormats/DataRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/DataRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/DataRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/DataRecord/interface/CSCAlignmentErrorRcd.h"


class MuonGeometryRecord : public edm::eventsetup::DependentRecordImplementation<MuonGeometryRecord,boost::mpl::vector<IdealGeometryRecord, DTAlignmentRcd, DTAlignmentErrorRcd, CSCAlignmentRcd, CSCAlignmentErrorRcd> > {};

#endif

