#include "CondFormats/ESObjects/interface/ESMissingEnergyCalibration.h"

ESMissingEnergyCalibration::ESMissingEnergyCalibration() 
{
  ConstAEta0_  = 0.;
  ConstBEta0_  = 0.;

  ConstAEta1_  = 0.;
  ConstBEta1_  = 0.;

  ConstAEta2_  = 0.;
  ConstBEta2_  = 0.;

  ConstAEta3_  = 0.;
  ConstBEta3_  = 0.;
}

ESMissingEnergyCalibration::ESMissingEnergyCalibration(
  const float & ConstAEta0, const float & ConstBEta0, 
  const float & ConstAEta1, const float & ConstBEta1, 
  const float & ConstAEta2, const float & ConstBEta2, 
  const float & ConstAEta3, const float & ConstBEta3
  ) 
{
  ConstAEta0_  = ConstAEta0;
  ConstBEta0_  = ConstBEta0;

  ConstAEta1_  = ConstAEta1;
  ConstBEta1_  = ConstBEta1;

  ConstAEta2_  = ConstAEta2;
  ConstBEta2_  = ConstBEta2;

  ConstAEta3_  = ConstAEta3;
  ConstBEta3_  = ConstBEta3;
}

ESMissingEnergyCalibration::~ESMissingEnergyCalibration() {

}
