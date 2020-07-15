#ifndef RecoMTD_Record_MTDRecoGeometryRecord_h
#define RecoMTD_Record_MTDRecoGeometryRecord_h

/** \class MTDRecoGeometryRecord
 *
 *  Record to hold mtd reconstruction geometries.
 *
 *  \author L. Gray - FNAL
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"

#include <boost/mp11/list.hpp>

class MTDRecoGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<MTDRecoGeometryRecord,
                                                            boost::mp11::mp_list<MTDDigiGeometryRecord> > {};

#endif
