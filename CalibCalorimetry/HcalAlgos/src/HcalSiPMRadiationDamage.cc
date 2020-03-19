#include "CalibCalorimetry/HcalAlgos/interface/HcalSiPMRadiationDamage.h"

#include <vector>
#include <cmath>

//default constructor
HcalSiPMRadiationDamage::HcalSiPMRadiationDamage()
    : temperatureBase_(0.0), temperatureNew_(0.0), intlumiOffset_(0.0), depVsTemp_(0.0), intlumiToNeutrons_(0.0) {}

HcalSiPMRadiationDamage::HcalSiPMRadiationDamage(std::vector<double> darkCurrentBase, const edm::ParameterSet& p)
    : darkCurrentBase_(darkCurrentBase),
      temperatureBase_(p.getParameter<double>("temperatureBase")),
      temperatureNew_(p.getParameter<double>("temperatureNew")),
      intlumiOffset_(p.getParameter<double>("intlumiOffset")),
      depVsTemp_(p.getParameter<double>("depVsTemp")),
      intlumiToNeutrons_(p.getParameter<double>("intlumiToNeutrons")),
      depVsNeutrons_(p.getParameter<std::vector<double>>("depVsNeutrons")) {}

//accounts for both lumi dependence and temp dependence
double HcalSiPMRadiationDamage::getDarkCurrent(double intlumi, unsigned index) const {
  intlumi -= intlumiOffset_;
  if (intlumi <= 0)
    return darkCurrentBase_.at(index);
  double darkCurrentNewLumi = darkCurrentBase_.at(index) + depVsNeutrons_.at(index) * (intlumi * intlumiToNeutrons_);
  double darkCurrentNewTemp = darkCurrentNewLumi * std::exp(depVsTemp_ * (temperatureNew_ - temperatureBase_));
  return darkCurrentNewTemp;
}
