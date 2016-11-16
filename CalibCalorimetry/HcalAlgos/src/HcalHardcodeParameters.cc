#include "CalibCalorimetry/HcalAlgos/interface/HcalHardcodeParameters.h"

HcalHardcodeParameters::HcalHardcodeParameters(double pedestal, double pedestalWidth, std::vector<double> gain, std::vector<double> gainWidth, 
											   int qieType, std::vector<double> qieOffset, std::vector<double> qieSlope, int mcShape, int recoShape,
											   double photoelectronsToAnalog, std::vector<double> darkCurrent)
:	pedestal_(pedestal),
	pedestalWidth_(pedestalWidth),
	gain_(gain),
	gainWidth_(gainWidth),
	qieType_(qieType),
	qieOffset_(qieOffset),
	qieSlope_(qieSlope),
	mcShape_(mcShape),
	recoShape_(recoShape),
	photoelectronsToAnalog_(photoelectronsToAnalog),
	darkCurrent_(darkCurrent)
{
}

HcalHardcodeParameters::HcalHardcodeParameters(const edm::ParameterSet & p)
:	pedestal_(p.getParameter<double>("pedestal")),
	pedestalWidth_(p.getParameter<double>("pedestalWidth")),
	gain_(p.getParameter<std::vector<double>>("gain")),
	gainWidth_(p.getParameter<std::vector<double>>("gainWidth")),
	qieType_(p.getParameter<int>("qieType")),
	qieOffset_(p.getParameter<std::vector<double>>("qieOffset")),
	qieSlope_(p.getParameter<std::vector<double>>("qieSlope")),
	mcShape_(p.getParameter<int>("mcShape")),
	recoShape_(p.getParameter<int>("recoShape")),
	photoelectronsToAnalog_(p.getParameter<double>("photoelectronsToAnalog")),
	darkCurrent_(p.getParameter<std::vector<double>>("darkCurrent"))
{
}
