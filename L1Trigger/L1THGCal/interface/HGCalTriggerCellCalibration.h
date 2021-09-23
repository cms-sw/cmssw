#ifndef __L1Trigger_L1THGCal_HGCalTriggerCellCalibration_h__
#define __L1Trigger_L1THGCal_HGCalTriggerCellCalibration_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalTriggerCellCalibration {
public:
  HGCalTriggerCellCalibration(const edm::ParameterSet& conf);
  void setGeometry(const HGCalTriggerGeometryBase* const geom) { triggerTools_.setGeometry(geom); }
  void calibrateInMipT(l1t::HGCalTriggerCell&) const;
  void calibrateMipTinGeV(l1t::HGCalTriggerCell&) const;
  void calibrateInGeV(l1t::HGCalTriggerCell&) const;

private:
  double lsb_;
  std::vector<double> fCperMIP_;
  std::vector<double> chargeCollectionEfficiency_;
  std::vector<double> thicknessCorrection_;
  std::vector<double> dEdX_weights_;

  HGCalTriggerTools triggerTools_;
};

#endif
