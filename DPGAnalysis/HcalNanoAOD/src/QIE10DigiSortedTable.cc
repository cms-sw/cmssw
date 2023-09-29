#include "DPGAnalysis/HcalNanoAOD/interface/QIE10DigiSortedTable.h"

QIE10DigiSortedTable::QIE10DigiSortedTable(const std::vector<HcalDetId>& dids, const unsigned int nTS) {
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
  linkErrors_.resize(dids_.size());
  flags_.resize(dids_.size());
  sois_.resize(dids_.size());
  valids_.resize(dids_.size());
  //sipmTypes_.resize(dids_.size());

  adcs_.resize(nTS_, std::vector<int>(dids_.size()));
  fcs_.resize(nTS_, std::vector<float>(dids_.size()));
  pedestalfcs_.resize(nTS_, std::vector<float>(dids_.size()));
  tdcs_.resize(nTS_, std::vector<int>(dids_.size()));
  capids_.resize(nTS_, std::vector<int>(dids_.size()));
  oks_.resize(nTS_, std::vector<bool>(dids_.size()));
}

void QIE10DigiSortedTable::add(const QIE10DataFrame* digi, const edm::ESHandle<HcalDbService>& dbService) {
  HcalDetId did = digi->detid();
  unsigned int index = did_indexmap_.at(did);

  CaloSamples digiCaloSamples = hcaldqm::utilities::loadADC2fCDB<QIE10DataFrame>(dbService, did, *digi);
  HcalCalibrations calibrations = dbService->getHcalCalibrations(did);

  ietas_[index] = did.ieta();
  iphis_[index] = did.iphi();
  subdets_[index] = did.subdet();
  depths_[index] = did.depth();
  rawIds_[index] = did.rawId();
  linkErrors_[index] = digi->linkError();
  flags_[index] = digi->flags();
  //sipmTypes_[index] = (uint8_t)dbService->getHcalSiPMParameter(did)->getType();

  for (unsigned int iTS = 0; iTS < (unsigned int)digi->samples(); ++iTS) {
    if ((*digi)[iTS].soi()) {
      sois_[index] = iTS;
    }
    oks_[iTS][index] = (*digi)[iTS].ok();
    adcs_[iTS][index] = (*digi)[iTS].adc();
    tdcs_[iTS][index] = (*digi)[iTS].le_tdc();
    //tetdcs_[iTS][index]    = (*digi)[iTS].te_tdc();
    capids_[iTS][index] = (*digi)[iTS].capid();
    fcs_[iTS][index] = digiCaloSamples[iTS];
    pedestalfcs_[iTS][index] = calibrations.pedestal((*digi)[iTS].capid());
  }
  valids_[index] = true;
}

void QIE10DigiSortedTable::reset() {
  std::fill(ietas_.begin(), ietas_.end(), 0);
  std::fill(iphis_.begin(), iphis_.end(), 0);
  std::fill(subdets_.begin(), subdets_.end(), 0);
  std::fill(depths_.begin(), depths_.end(), 0);
  std::fill(rawIds_.begin(), rawIds_.end(), 0);
  std::fill(linkErrors_.begin(), linkErrors_.end(), 0);
  std::fill(flags_.begin(), flags_.end(), 0);
  std::fill(sois_.begin(), sois_.end(), -1);
  std::fill(valids_.begin(), valids_.end(), false);
  //std::fill(sipmTypes_.begin(), sipmTypes_.end(), 0);

  for (unsigned int i = 0; i < nTS_; ++i) {
    for (unsigned int j = 0; j < dids_.size(); ++j) {
      adcs_[i][j] = 0;
      fcs_[i][j] = 0.;
      pedestalfcs_[i][j] = 0.;
      tdcs_[i][j] = 0;
      capids_[i][j] = 0;
      oks_[i][j] = false;
    }
  }
}
