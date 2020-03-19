#ifndef GEOMETRY_RECORDS_MUON_GEOMETRY_RCD_H
#define GEOMETRY_RECORDS_MUON_GEOMETRY_RCD_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/DDSpecParRegistryRcd.h"
#include "Geometry/Records/interface/MuonNumberingRcd.h"
#include "Geometry/Records/interface/DTRecoGeometryRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "boost/mpl/vector.hpp"

class MuonGeometryRcd
    : public edm::eventsetup::DependentRecordImplementation<MuonGeometryRcd,
                                                            boost::mpl::vector<MuonNumberingRcd,
                                                                               DDSpecParRegistryRcd,
                                                                               GlobalPositionRcd,
                                                                               DTAlignmentRcd,
                                                                               DTAlignmentErrorRcd,
                                                                               DTAlignmentErrorExtendedRcd,
                                                                               DTRecoGeometryRcd>> {};
#endif
