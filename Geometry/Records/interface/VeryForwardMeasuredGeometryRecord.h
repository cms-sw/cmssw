/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*	Dominik Mierzejewski (dmierzej@cern.ch)
*
****************************************************************************/

#ifndef RECORDS_VeryForwardMeasuredGeometryRecord_H
#define RECORDS_VeryForwardMeasuredGeometryRecord_H

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "boost/mpl/vector.hpp"

#include "CondFormats/AlignmentRecord/interface/RPMeasuredAlignmentRecord.h"

/**
 * \ingroup TotemRPGeometry
 * \brief Event setup record containing the Measured (measured) geometry information.
 **/
class VeryForwardMeasuredGeometryRecord : public edm::eventsetup::DependentRecordImplementation
						   <VeryForwardMeasuredGeometryRecord, boost::mpl::vector<IdealGeometryRecord, RPMeasuredAlignmentRecord /*, ... */> >
{
};

#endif

