#ifndef CALIBTRACKER_SISTRIPCONNECTIVITY_SISTRIPREGIONCABLINGRCD_H
#define CALIBTRACKER_SISTRIPCONNECTIVITY_SISTRIPREGIONCABLINGRCD_H

/** Class : SiStripRegionCablingRcd SiStripRegionCablingRcd.h CalibTracker/SiStripConnectivity/plugins/SiStripRegionCablingRcd.h

    Author : pwing
*/



#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "boost/mpl/vector.hpp"

class SiStripRegionCablingRcd : public edm::eventsetup::DependentRecordImplementation<SiStripRegionCablingRcd,
  boost::mpl::vector<SiStripDetCablingRcd,TrackerDigiGeometryRecord> > {};

#endif /* CALIBTRACKER_RECORDS_SISTRIPREGIONCABLINGRCD_H */

