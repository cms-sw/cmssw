#include "DPGAnalysis/HcalNanoAOD/interface/HODigiSortedTable.h"

HODigiSortedTable::HODigiSortedTable(const std::vector<HcalDetId>& dids, const unsigned int nTS) {
  dids_ = dids;
  for (std::vector<HcalDetId>::const_iterator it_did = dids_.begin(); it_did != dids_.end(); ++it_did) {
    did_indexmap_[*it_did] = (unsigned int)(it_did - dids_.begin());
  }

  nTS_ = nTS;

  ietas_.resize(dids_.size());
  iphis_.resize(dids_.size());
  subdets_.resize(dids_.size());
  depths_.resize(dids_.size());
  rawIds_.resize(dids_.size());
  fiberIdleOffsets_.resize(dids_.size());
  sois_.resize(dids_.size());
  valids_.resize(dids_.size());

  adcs_.resize(nTS_, std::vector<int>(dids_.size()));
  fcs_.resize(nTS_, std::vector<float>(dids_.size()));
  pedestalfcs_.resize(nTS_, std::vector<float>(dids_.size()));
  capids_.resize(nTS_, std::vector<int>(dids_.size()));
  fibers_.resize(nTS_, std::vector<int>(dids_.size()));
  fiberChans_.resize(nTS_, std::vector<int>(dids_.size()));
  dvs_.resize(nTS_, std::vector<int>(dids_.size()));
  ers_.resize(nTS_, std::vector<int>(dids_.size()));
}

void HODigiSortedTable::add(const HODataFrame* digi, const edm::ESHandle<HcalDbService>& dbService) {
  HcalDetId did = digi->id();
  unsigned int index = did_indexmap_.at(did);

  CaloSamples digiCaloSamples = hcaldqm::utilities::loadADC2fCDB<HODataFrame>(dbService, did, *digi);
  HcalCalibrations calibrations = dbService->getHcalCalibrations(did);

  ietas_[index] = did.ieta();
  iphis_[index] = did.iphi();
  subdets_[index] = did.subdet();
  depths_[index] = did.depth();
  rawIds_[index] = did.rawId();
  fiberIdleOffsets_[index] = digi->fiberIdleOffset();
  sois_[index] = digi->presamples();

  for (unsigned int iTS = 0; iTS < (unsigned int)digi->size(); ++iTS) {
    adcs_[iTS][index] = digi->sample(iTS).adc();
    capids_[iTS][index] = digi->sample(iTS).capid();
    fcs_[iTS][index] = digiCaloSamples[iTS];
    pedestalfcs_[iTS][index] = calibrations.pedestal(digi->sample(iTS).capid());
    dvs_[iTS][index] = digi->sample(iTS).dv();
    ers_[iTS][index] = digi->sample(iTS).er();
  }
  valids_[index] = true;
}

void HODigiSortedTable::reset() {
  std::fill(ietas_.begin(), ietas_.end(), 0);
  std::fill(iphis_.begin(), iphis_.end(), -100);
  std::fill(subdets_.begin(), subdets_.end(), -1);
  std::fill(depths_.begin(), depths_.end(), 0);
  std::fill(rawIds_.begin(), rawIds_.end(), 0);
  std::fill(fiberIdleOffsets_.begin(), fiberIdleOffsets_.end(), 0);
  std::fill(sois_.begin(), sois_.end(), -1);
  std::fill(valids_.begin(), valids_.end(), false);

  for (unsigned int i = 0; i < nTS_; ++i) {
    for (unsigned int j = 0; j < dids_.size(); ++j) {
      adcs_[i][j] = 0;
      fcs_[i][j] = 0;
      pedestalfcs_[i][j] = 0;
      capids_[i][j] = 0;
      fibers_[i][j] = 0;
      fiberChans_[i][j] = 0;
      dvs_[i][j] = 0;
      ers_[i][j] = 0;
    }
  }
}
