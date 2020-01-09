#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorSuperTriggerCellImpl.h"

HGCalConcentratorSuperTriggerCellImpl::HGCalConcentratorSuperTriggerCellImpl(const edm::ParameterSet& conf)
    : fixedDataSizePerHGCROC_(conf.getParameter<bool>("fixedDataSizePerHGCROC")),
      coarsenTriggerCells_(conf.getParameter<bool>("coarsenTriggerCells")),
      coarseTCmapping_(conf.getParameter<std::vector<unsigned>>("ctcSize")),
      superTCmapping_(conf.getParameter<std::vector<unsigned>>("stcSize")),
      calibration_(conf.getParameterSet("superTCCalibration")),
      vfeCompression_(conf.getParameterSet("superTCCompression")) {
  std::string energyType(conf.getParameter<string>("type_energy_division"));

  if (energyType == "superTriggerCell") {
    energyDivisionType_ = superTriggerCell;
  } else if (energyType == "oneBitFraction") {
    energyDivisionType_ = oneBitFraction;

    oneBitFractionThreshold_ = conf.getParameter<double>("oneBitFractionThreshold");
    oneBitFractionLowValue_ = conf.getParameter<double>("oneBitFractionLowValue");
    oneBitFractionHighValue_ = conf.getParameter<double>("oneBitFractionHighValue");

  } else if (energyType == "equalShare") {
    energyDivisionType_ = equalShare;

  } else {
    energyDivisionType_ = superTriggerCell;
  }
}

uint32_t HGCalConcentratorSuperTriggerCellImpl::getCompressedSTCEnergy(const SuperTriggerCell& stc) const {
  uint32_t code(0);
  uint32_t compressed_value(0);
  vfeCompression_.compressSingle(stc.getSumHwPt(), code, compressed_value);
  return compressed_value;
}

void HGCalConcentratorSuperTriggerCellImpl::createAllTriggerCells(
    std::unordered_map<unsigned, SuperTriggerCell>& STCs, std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput) const {
  for (auto& s : STCs) {
    std::vector<uint32_t> output_ids = superTCmapping_.getConstituentTriggerCells(s.second.getSTCId());

    int thickness = (!output_ids.empty() ? triggerTools_.thicknessIndex(output_ids.at(0), true) : 0);

    for (const auto& id : output_ids) {
      if (fixedDataSizePerHGCROC_ && thickness > kHighDensityThickness_ &&
          id != coarseTCmapping_.getRepresentativeDetId(id)) {
        continue;
      }

      if (!triggerTools_.getTriggerGeometry()->validTriggerCell(id)) {
        continue;
      }

      l1t::HGCalTriggerCell triggerCell;
      triggerCell.setDetId(id);
      if (energyDivisionType_ == superTriggerCell && id != s.second.getMaxId()) {
        continue;
      }

      DetId tc_Id(id);

      //To guard against the case in v8 geometry where
      //there might be different thicknesses within a module
      //This is a small effect, but in principle energy might
      //be lost.
      if (superTCmapping_.getCoarseTriggerCellId(id) != s.second.getSTCId()) {
        if (triggerTools_.getTriggerGeometry()->isV9Geometry()) {
          throw cms::Exception("NonExistingCoarseTC")
              << "The coarse trigger cell correponsing to the nominal trigger cell does not exist";
        } else {
          continue;
        }
      }

      trigCellVecOutput.push_back(triggerCell);

      if (energyDivisionType_ == oneBitFraction) {  //Get the 1 bit fractions

        if (id != s.second.getMaxId()) {
          float tc_fraction = getTriggerCellOneBitFraction(s.second.getTCpt(id), s.second.getSumPt());
          s.second.addToFractionSum(tc_fraction);
        }
      }
    }
  }

  // assign energy
  for (l1t::HGCalTriggerCell& tc : trigCellVecOutput) {
    const auto& stc = STCs[superTCmapping_.getCoarseTriggerCellId(tc.detId())];
    assignSuperTriggerCellEnergyAndPosition(tc, stc);
  }
}

void HGCalConcentratorSuperTriggerCellImpl::assignSuperTriggerCellEnergyAndPosition(l1t::HGCalTriggerCell& c,
                                                                                    const SuperTriggerCell& stc) const {
  //Compress and recalibrate STC energy
  uint32_t compressed_value = getCompressedSTCEnergy(stc);

  int thickness = triggerTools_.thicknessIndex(c.detId(), true);

  GlobalPoint point;
  if ((fixedDataSizePerHGCROC_ && thickness > kHighDensityThickness_) || coarsenTriggerCells_) {
    point = coarseTCmapping_.getCoarseTriggerCellPosition(coarseTCmapping_.getCoarseTriggerCellId(c.detId()));
  } else {
    point = triggerTools_.getTCPosition(c.detId());
  }
  c.setPosition(point);

  math::PtEtaPhiMLorentzVector p4(c.pt(), point.eta(), point.phi(), 0.);
  c.setP4(p4);

  if (energyDivisionType_ == superTriggerCell) {
    if (c.detId() == stc.getMaxId()) {
      c.setHwPt(compressed_value);
      calibration_.calibrateInGeV(c);
    } else {
      throw cms::Exception("NonMaxIdSuperTriggerCell")
          << "Trigger Cell with detId not equal to the maximum of the superTriggerCell found";
    }
  } else if (energyDivisionType_ == equalShare) {
    double coarseTriggerCellSize =
        coarsenTriggerCells_
            ? double(
                  coarseTCmapping_.getConstituentTriggerCells(coarseTCmapping_.getCoarseTriggerCellId(stc.getMaxId()))
                      .size())
            : 1.;

    double denominator =
        fixedDataSizePerHGCROC_
            ? double(kTriggerCellsForDivision_)
            : double(superTCmapping_.getConstituentTriggerCells(stc.getSTCId()).size()) / coarseTriggerCellSize;

    c.setHwPt(std::round(compressed_value / denominator));
    calibration_.calibrateInGeV(c);

  } else if (energyDivisionType_ == oneBitFraction) {
    double frac = 0;

    if (c.detId() != stc.getMaxId()) {
      frac = getTriggerCellOneBitFraction(stc.getTCpt(c.detId()), stc.getSumPt());
    } else {
      frac = 1 - stc.getFractionSum();
    }

    c.setHwPt(std::round(compressed_value * frac));
    calibration_.calibrateInGeV(c);
  }
}

float HGCalConcentratorSuperTriggerCellImpl::getTriggerCellOneBitFraction(float tcPt, float sumPt) const {
  double f = tcPt / sumPt;
  double frac = 0;
  if (f < oneBitFractionThreshold_) {
    frac = oneBitFractionLowValue_;
  } else {
    frac = oneBitFractionHighValue_;
  }

  return frac;
}

void HGCalConcentratorSuperTriggerCellImpl::select(const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput,
                                                   std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput) {
  std::unordered_map<unsigned, SuperTriggerCell> STCs;
  // first pass, fill the "coarse" trigger cells
  for (const l1t::HGCalTriggerCell& tc : trigCellVecInput) {
    uint32_t stcid = superTCmapping_.getCoarseTriggerCellId(tc.detId());
    STCs[stcid].add(tc, stcid);
  }

  createAllTriggerCells(STCs, trigCellVecOutput);
}
