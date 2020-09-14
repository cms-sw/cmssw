/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*	Jan Kaspar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#ifndef RECORDS_VeryForwardRealGeometryRecord_H
#define RECORDS_VeryForwardRealGeometryRecord_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "FWCore/Utilities/interface/mplVector.h"

#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"

/**
 * \ingroup TotemRPGeometry
 * \brief Event setup record containing the real (actual) geometry information.
 **/
class VeryForwardRealGeometryRecord : public edm::eventsetup::DependentRecordImplementation<
                                          VeryForwardRealGeometryRecord,
                                          edm::mpl::Vector<IdealGeometryRecord, RPRealAlignmentRecord /*, ... */> > {};

#endif
