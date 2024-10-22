/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*	Jan Kaspar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#ifndef RECORDS_VeryForwardMisalignedGeometryRecord_H
#define RECORDS_VeryForwardMisalignedGeometryRecord_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardIdealGeometryRecord.h"

#include "FWCore/Utilities/interface/mplVector.h"

#include "CondFormats/AlignmentRecord/interface/RPMisalignedAlignmentRecord.h"

/**
 * \ingroup TotemRPGeometry
 * \brief Event setup record containing the misaligned geometry information. It is used for 
 * alignment studies only.
 **/
class VeryForwardMisalignedGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<
          VeryForwardMisalignedGeometryRecord,
          edm::mpl::Vector<VeryForwardIdealGeometryRecord, IdealGeometryRecord, RPMisalignedAlignmentRecord /*, ... */> > {
};

#endif
