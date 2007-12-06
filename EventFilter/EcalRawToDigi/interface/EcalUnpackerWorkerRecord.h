#ifndef EcalUnpackerWorkerRecord_H
#define EcalUnpackerWorkerRecord_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"

#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "EventFilter/EcalRawToDigi/interface/EcalRegionCablingRecord.h"

class EcalUnpackerWorkerRecord : public edm::eventsetup::DependentRecordImplementation<EcalUnpackerWorkerRecord,
  boost::mpl::vector<EcalPedestalsRcd,
  EcalGainRatiosRcd,
  EcalWeightXtalGroupsRcd,
  EcalTBWeightsRcd,
  EcalIntercalibConstantsRcd,
  EcalADCToGeVConstantRcd,
  EcalLaserDbRecord,
  EcalRegionCablingRecord> > {};
#endif
