#include "CalibCalorimetry/EcalTPGTools/interface/EcalReadoutTools.h"

EcalReadoutTools::EcalReadoutTools(const edm::Event &iEvent, const edm::EventSetup &iSetup) {

  edm::ESHandle<EcalTrigTowerConstituentsMap> hTriggerTowerMap;
  iSetup.get<IdealGeometryRecord>().get(hTriggerTowerMap);
  triggerTowerMap_ = hTriggerTowerMap.product(); 
  
  edm::ESHandle< EcalElectronicsMapping > ecalmapping;
  iSetup.get< EcalMappingRcd >().get(ecalmapping);
  elecMap_ = ecalmapping.product();

}

EcalTrigTowerDetId EcalReadoutTools::readOutUnitOf(const EBDetId& xtalId) const{
  return triggerTowerMap_->towerOf(xtalId);
}

EcalScDetId EcalReadoutTools::readOutUnitOf(const EEDetId& xtalId) const{
  const EcalElectronicsId& EcalElecId = elecMap_->getElectronicsId(xtalId);
  int iDCC= EcalElecId.dccId();
  int iDccChan = EcalElecId.towerId();
  const bool ignoreSingle = true;
  const std::vector<EcalScDetId> id = elecMap_->getEcalScDetId(iDCC, iDccChan, ignoreSingle);
  return id.size()>0?id[0]:EcalScDetId();
}
