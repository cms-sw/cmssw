#include "DPGAnalysis/HcalNanoAOD/interface/HFPreRecHitSortedTable.h"

HFPreRecHitSortedTable::HFPreRecHitSortedTable(const std::vector<HcalDetId>& dids) {
  dids_ = dids;
  for (std::vector<HcalDetId>::const_iterator it_did = dids_.begin(); it_did != dids_.end(); ++it_did) {
    did_indexmap_[*it_did] = (unsigned int)(it_did - dids_.begin());
  }

  charges_.resize(dids_.size());
  chargeAsymmetries_.resize(dids_.size());
  valids_.resize(dids_.size());
}

void HFPreRecHitSortedTable::add(const HFPreRecHitCollection::const_iterator itPreRecHit) {
  HcalDetId did = itPreRecHit->id();
  unsigned int index = did_indexmap_.at(did);

  charges_[index] = itPreRecHit->charge();
  chargeAsymmetries_[index] =
      itPreRecHit->chargeAsymmetry(0.).first;  // chargeAsymmetry() returns std::pair<float qAsym, bool passCut>
  valids_[index] = true;
}

void HFPreRecHitSortedTable::reset() {
  std::fill(charges_.begin(), charges_.end(), 0);
  std::fill(chargeAsymmetries_.begin(), chargeAsymmetries_.end(), 0);
  std::fill(valids_.begin(), valids_.end(), false);
}
