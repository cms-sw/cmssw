#include "L1Trigger/L1THGCal/interface/HGCalTriggerCellCalibration.h"

#include <cmath>

HGCalTriggerCellCalibration::HGCalTriggerCellCalibration(const edm::ParameterSet& conf)
    : lsb_(conf.getParameter<double>("lsb")),
      fCperMIP_(conf.getParameter<std::vector<double>>("fCperMIP")),
      chargeCollectionEfficiency_(conf.getParameter<edm::ParameterSet>("chargeCollectionEfficiency")
                                      .getParameter<std::vector<double>>("values")),
      thicknessCorrection_(conf.getParameter<std::vector<double>>("thicknessCorrection")),
      dEdX_weights_(conf.getParameter<std::vector<double>>("dEdXweights")) {
  for (const auto& fCperMIP : fCperMIP_) {
    if (fCperMIP <= 0) {
      edm::LogWarning("DivisionByZero") << "WARNING: zero or negative MIP->fC correction factor. It won't be "
                                           "applied to correct trigger cell energies.";
    }
  }
  for (const auto& cce : chargeCollectionEfficiency_) {
    if (cce <= 0) {
      edm::LogWarning("DivisionByZero") << "WARNING: zero or negative cell-thickness correction factor. It won't be "
                                           "applied to correct trigger cell energies.";
    }
  }
  for (const auto& thickCorr : thicknessCorrection_) {
    if (thickCorr <= 0) {
      edm::LogWarning("DivisionByZero") << "WARNING: zero or negative cell-thickness correction factor. It won't be "
                                           "applied to correct trigger cell energies.";
    }
  }
}

void HGCalTriggerCellCalibration::calibrateInMipT(l1t::HGCalTriggerCell& trgCell) const {
  DetId trgdetid(trgCell.detId());
  bool isSilicon = triggerTools_.isSilicon(trgdetid);
  constexpr int kScintillatorIndex = 0;
  unsigned thickness = isSilicon ? triggerTools_.thicknessIndex(trgdetid) : kScintillatorIndex;
  if (thickness >= fCperMIP_.size()) {
    throw cms::Exception("OutOfBound") << "Trying to access thickness index " << thickness
                                       << " in fCperMIP, which is of size " << fCperMIP_.size();
  }
  if (thickness >= chargeCollectionEfficiency_.size()) {
    throw cms::Exception("OutOfBound") << "Trying to access thickness index " << thickness
                                       << " in chargeCollectionEfficiency, which is of size "
                                       << chargeCollectionEfficiency_.size();
  }

  /* get the hardware pT in ADC counts: */
  int hwPt = trgCell.hwPt();

  // Convert ADC to charge in fC (in EE+FH) or in MIPs (in BH)
  double amplitude = hwPt * lsb_;

  if (chargeCollectionEfficiency_[thickness] > 0) {
    amplitude /= chargeCollectionEfficiency_[thickness];
  }

  /* convert the charge amplitude in MIP: */
  double trgCellMipP = amplitude;

  if (fCperMIP_[thickness] > 0) {
    trgCellMipP /= fCperMIP_[thickness];
  }

  /* compute the transverse-mip */
  double trgCellMipPt = trgCellMipP / std::cosh(trgCell.eta());

  /* setting pT [mip] */
  trgCell.setMipPt(trgCellMipPt);
}

void HGCalTriggerCellCalibration::calibrateMipTinGeV(l1t::HGCalTriggerCell& trgCell) const {
  constexpr double MevToGeV(0.001);
  double trgCellEt(0.);

  DetId trgdetid(trgCell.detId());
  unsigned trgCellLayer = triggerTools_.layerWithOffset(trgdetid);
  bool isSilicon = triggerTools_.isSilicon(trgdetid);
  constexpr int kScintillatorIndex = 0;
  unsigned thickness = isSilicon ? triggerTools_.thicknessIndex(trgdetid) : kScintillatorIndex;
  if (thickness >= thicknessCorrection_.size()) {
    throw cms::Exception("OutOfBound") << "Trying to access thickness index " << thickness
                                       << " in thicknessCorrection, which is of size " << thicknessCorrection_.size();
  }

  /* weight the amplitude by the absorber coefficient in MeV/mip + bring it in
   * GeV */
  trgCellEt = trgCell.mipPt() * MevToGeV;
  trgCellEt *= dEdX_weights_.at(trgCellLayer);

  /* correct for the cell-thickness */
  if (thicknessCorrection_[thickness] > 0) {
    trgCellEt /= thicknessCorrection_[thickness];
  }

  math::PtEtaPhiMLorentzVector calibP4(trgCellEt, trgCell.eta(), trgCell.phi(), 0.);
  trgCell.setP4(calibP4);
}

void HGCalTriggerCellCalibration::calibrateInGeV(l1t::HGCalTriggerCell& trgCell) const {
  /* calibrate from ADC count to transverse mip */
  calibrateInMipT(trgCell);

  /* calibrate from mip count to GeV */
  calibrateMipTinGeV(trgCell);
}
