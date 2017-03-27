#ifndef CalibCalorimetry_HcalSiPMRadiationDamage_h
#define CalibCalorimetry_HcalSiPMRadiationDamage_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HcalCalibObjects/interface/HBHEDarkening.h"

#include <vector>

// SiPM radiation damage model for HB and HE
// neutron fluence [cm-2] (~ radiation dose [Mrad] ~ integrated luminosity [fb-1]) increases dark current [uA] (linearly)
// decrease in temperature [°C] decreases dark current (exponentially)
// neutron fluence at HB and HE RBX locations modeled with Fluka

class HcalSiPMRadiationDamage {
	public:
		HcalSiPMRadiationDamage();
		HcalSiPMRadiationDamage(std::vector<double> darkCurrentBase_, const edm::ParameterSet & p);
		~HcalSiPMRadiationDamage() {}
		
		//accessors
		double getDarkCurrent(double intlumi, unsigned index) const;

	private:
		//members
		std::vector<double> darkCurrentBase;
		double temperatureBase;
		double temperatureNew;
		double intlumiOffset;
		double depVsTemp;
		double intlumiToNeutrons;
		std::vector<double> depVsNeutrons;
};

#endif // HBHERecalibration_h
