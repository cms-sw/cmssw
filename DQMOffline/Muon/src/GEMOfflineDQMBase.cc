#include "DQMOffline/Muon/interface/GEMOfflineDQMBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

GEMOfflineDQMBase::GEMOfflineDQMBase(const edm::ParameterSet& pset) {
  log_category_ = pset.getUntrackedParameter<std::string>("logCategory");
}

int GEMOfflineDQMBase::getDetOccXBin(const GEMDetId& gem_id, const edm::ESHandle<GEMGeometry>& gem) {
  const GEMSuperChamber* superchamber = gem->superChamber(gem_id);
  if (superchamber == nullptr) {
    return -1;
  }
  return getDetOccXBin(gem_id.chamber(), gem_id.layer(), superchamber->nChambers());
}

void GEMOfflineDQMBase::setDetLabelsVFAT(MonitorElement* me, const GEMStation* station) {
  if (me == nullptr) {
    edm::LogError(log_category_) << "MonitorElement* is nullptr" << std::endl;
    return;
  }

  me->setAxisTitle("Superchamber / Chamber", 1);
  for (const GEMSuperChamber* superchamber : station->superChambers()) {
    const int num_chambers = superchamber->nChambers();
    for (const GEMChamber* chamber : superchamber->chambers()) {
      const int sc = chamber->id().chamber();
      const int ch = chamber->id().layer();
      const int xbin = getDetOccXBin(sc, ch, num_chambers);
      const char* label = Form("%d/%d", sc, ch);
      me->setBinLabel(xbin, label, 1);
    }
  }

  me->setAxisTitle("VFAT (i#eta)", 2);
  const int max_vfat = getMaxVFAT(station->station());
  if (max_vfat < 0) {
    edm::LogError(log_category_) << "Wrong max VFAT: " << max_vfat << " at Station " << station->station() << std::endl;
    return;
  }

  for (int ieta = 1; ieta <= GEMeMap::maxEtaPartition_; ieta++) {
    for (int vfat_phi = 1; vfat_phi <= max_vfat; vfat_phi++) {
      const int ybin = getVFATNumber(station->station(), ieta, vfat_phi);
      const char* label = Form("%d (%d)", ybin, ieta);
      me->setBinLabel(ybin, label, 2);
    }
  }
}

void GEMOfflineDQMBase::setDetLabelsEta(MonitorElement* me, const GEMStation* station) {
  if (me == nullptr) {
    edm::LogError(log_category_) << "MonitorElement* is nullptr" << std::endl;
    return;
  }

  me->setAxisTitle("Superchamber / Chamber", 1);
  for (const GEMSuperChamber* superchamber : station->superChambers()) {
    const int num_chambers = superchamber->nChambers();

    for (const GEMChamber* chamber : superchamber->chambers()) {
      const int sc = chamber->id().chamber();
      const int ch = chamber->id().layer();
      const int xbin = getDetOccXBin(sc, ch, num_chambers);
      const char* label = Form("%d/%d", sc, ch);
      me->setBinLabel(xbin, label, 1);
    }
  }

  me->setAxisTitle("i#eta", 2);
  for (int ieta = 1; ieta <= GEMeMap::maxEtaPartition_; ieta++) {
    const std::string&& label = std::to_string(ieta);
    me->setBinLabel(ieta, label, 2);
  }
}
