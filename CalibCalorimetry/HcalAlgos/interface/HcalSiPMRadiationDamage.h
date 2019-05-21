#ifndef CalibCalorimetry_HcalSiPMRadiationDamage_h
#define CalibCalorimetry_HcalSiPMRadiationDamage_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/HcalObjects/interface/HBHEDarkening.h"

#include <vector>

// SiPM radiation damage model for HB and HE
// neutron fluence [cm-2] (~ radiation dose [Mrad] ~ integrated luminosity [fb-1]) increases dark current [uA] (linearly)
// decrease in temperature [°C] decreases dark current (exponentially)
// neutron fluence at HB and HE RBX locations modeled with Fluka

class HcalSiPMRadiationDamage {
public:
  HcalSiPMRadiationDamage();
  HcalSiPMRadiationDamage(std::vector<double> darkCurrentBase, const edm::ParameterSet& p);
  ~HcalSiPMRadiationDamage() {}

  //accessors
  double getDarkCurrent(double intlumi, unsigned index) const;

private:
  //members
  std::vector<double> darkCurrentBase_;
  double temperatureBase_;
  double temperatureNew_;
  double intlumiOffset_;
  double depVsTemp_;
  double intlumiToNeutrons_;
  std::vector<double> depVsNeutrons_;
};

#endif  // HBHERecalibration_h
