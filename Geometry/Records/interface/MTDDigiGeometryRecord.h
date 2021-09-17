#ifndef RECORDS_MTDDIGIGEOMETRYRECORD_H
#define RECORDS_MTDDIGIGEOMETRYRECORD_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/AlignmentRecord/interface/MTDAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/MTDAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/MTDSurfaceDeformationRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/Records/interface/PMTDParametersRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

class MTDDigiGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<MTDDigiGeometryRecord,
                                                            edm::mpl::Vector<IdealGeometryRecord,
                                                                             MTDAlignmentRcd,
                                                                             MTDAlignmentErrorExtendedRcd,
                                                                             MTDSurfaceDeformationRcd,
                                                                             GlobalPositionRcd,
                                                                             MTDTopologyRcd,
                                                                             PMTDParametersRcd> > {};

#endif /* RECORDS_MTDDIGIGEOMETRYRECORD_H */
