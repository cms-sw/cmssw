#include "CalibCalorimetry/EcalTPGTools/interface/EcalReadoutTools.h"

EcalReadoutTools::EcalReadoutTools(const edm::Event&, const edm::EventSetup& iSetup, const ESGetTokens& esGetTokens) {
  triggerTowerMap_ = &iSetup.getData(esGetTokens.ecalTrigTowerConstituentsMapToken);
  elecMap_ = &iSetup.getData(esGetTokens.ecalElectronicsMappingToken);
}

EcalTrigTowerDetId EcalReadoutTools::readOutUnitOf(const EBDetId& xtalId) const {
  return triggerTowerMap_->towerOf(xtalId);
}

EcalScDetId EcalReadoutTools::readOutUnitOf(const EEDetId& xtalId) const {
  const EcalElectronicsId& EcalElecId = elecMap_->getElectronicsId(xtalId);
  int iDCC = EcalElecId.dccId();
  int iDccChan = EcalElecId.towerId();
  const bool ignoreSingle = true;
  const std::vector<EcalScDetId> id = elecMap_->getEcalScDetId(iDCC, iDccChan, ignoreSingle);
  return !id.empty() ? id[0] : EcalScDetId();
}
