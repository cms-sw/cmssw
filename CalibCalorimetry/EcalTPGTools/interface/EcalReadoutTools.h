#ifndef CalibCalorimetry_EcalTPGTools_EcalReadoutTools_H
#define CalibCalorimetry_EcalTPGTools_EcalReadoutTools_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

class EcalReadoutTools {

 private:
  const EcalTrigTowerConstituentsMap * triggerTowerMap_;
  const EcalElectronicsMapping* elecMap_;

 public:
  EcalReadoutTools(const edm::Event &iEvent, const edm::EventSetup &iSetup);
  EcalReadoutTools(const EcalReadoutTools&) = delete;
  EcalReadoutTools& operator=(const EcalReadoutTools&) = delete;

  EcalTrigTowerDetId readOutUnitOf(const EBDetId& xtalId) const;
  EcalScDetId        readOutUnitOf(const EEDetId& xtalId) const;
};

#endif
