#include "CalibCalorimetry/HcalAlgos/interface/HcalSiPMRadiationDamage.h"

#include <vector>
#include <cmath>

//default constructor
HcalSiPMRadiationDamage::HcalSiPMRadiationDamage() :
	temperatureBase(0.0), temperatureNew(0.0), intlumiOffset(0.0), depVsTemp(0.0), intlumiToNeutrons(0.0)
{}

HcalSiPMRadiationDamage::HcalSiPMRadiationDamage(std::vector<double> darkCurrentBase_, const edm::ParameterSet & p) :
	darkCurrentBase(darkCurrentBase_),
	temperatureBase(p.getParameter<double>("temperatureBase")),
	temperatureNew(p.getParameter<double>("temperatureNew")),
	intlumiOffset(p.getParameter<double>("intlumiOffset")),
	depVsTemp(p.getParameter<double>("depVsTemp")),
	intlumiToNeutrons(p.getParameter<double>("intlumiToNeutrons")),
	depVsNeutrons(p.getParameter<std::vector<double>>("depVsNeutrons"))
{}

//accounts for both lumi dependence and temp dependence
double HcalSiPMRadiationDamage::getDarkCurrent(double intlumi, unsigned index) const {
	intlumi -= intlumiOffset;
	if(intlumi<=0) return darkCurrentBase.at(index);
	double darkCurrentNewLumi = darkCurrentBase.at(index) + depVsNeutrons.at(index)*(intlumi*intlumiToNeutrons);
	double darkCurrentNewTemp = darkCurrentNewLumi*std::exp(depVsTemp*(temperatureNew - temperatureBase));
	return darkCurrentNewTemp;
}
