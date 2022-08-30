/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*	  Laurent Forthomme
*
****************************************************************************/

#ifndef Geometry_Records_TotemGeometryRcd_h
#define Geometry_Records_TotemGeometryRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PTotemRcd.h"

/**
 * \ingroup TotemGeometry
 * \brief Event setup record containing the real (actual) geometry information.
 **/
class TotemGeometryRcd
    : public edm::eventsetup::DependentRecordImplementation<TotemGeometryRcd,
                                                            edm::mpl::Vector<IdealGeometryRecord, PTotemRcd> > {};

#endif
