#ifndef Records_MuonGeometryRecord_h
#define Records_MuonGeometryRecord_h

/** \class MuonGeometryRecord
 *  The Muon DetUnit geometry.
 *
 *  $Date: 2012/11/15 23:02:26 $
 *  $Revision: 1.9 $
 *  \author N. Amapane - CERN
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/Records/interface/GEMRecoGeometryRcd.h"
#include "Geometry/Records/interface/RPCRecoGeometryRcd.h"
#include "Geometry/Records/interface/CSCRecoGeometryRcd.h"
#include "Geometry/Records/interface/DTRecoGeometryRcd.h"
#include "Geometry/Records/interface/CSCRecoDigiParametersRcd.h"
#include "boost/mpl/vector.hpp"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"

class MuonGeometryRecord : public edm::eventsetup::DependentRecordImplementation<MuonGeometryRecord,boost::mpl::vector<IdealGeometryRecord, MuonNumberingRecord, DTAlignmentRcd, DTAlignmentErrorRcd, CSCAlignmentRcd, CSCAlignmentErrorRcd, GlobalPositionRcd, GEMRecoGeometryRcd, RPCRecoGeometryRcd, DTRecoGeometryRcd, CSCRecoGeometryRcd, CSCRecoDigiParametersRcd> > {};

#endif

