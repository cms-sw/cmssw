#ifndef ValueAxis_h
#define ValueAxis_h

/*
 *	file:		ValueAxis.h
 *	Author:		Viktor Khristenko
 *
 *	Description:
 *		A wrapper around any axis that plots different value variables.
 *		No specific mapping between bins and values so far. 
 *
 *		Retrieving the bin number (for LS for instance) is hardcoded.
 */

#include "DQM/HcalCommon/interface/Axis.h"

namespace hcaldqm
{
	namespace axis
	{
		enum ValueType
		{
			fEntries = 0,
			fEvents = 1,
			f1OverEvents = 2,
			fEnergy = 3,
			fTime = 4,
			fADC = 5,
			fADC_5 = 6,
			fADC_15 = 7,
			fNomFC = 8,
			fNomFC_1000 = 9,
			fNomFC_3000 = 10,
			fTimeTS = 11,
			fTimeTS_200 = 12,
			fLS = 13,
			fEt_256 = 14,
			fEt_128 = 15,
			fFG = 16, 
			fRatio = 17,
			fDigiSize = 18,
			fAroundZero = 19,
			fRatio2 = 20,
			fEntries500 = 21,
			fEntries100 = 22,
			fdEtRatio = 23,
			fSumdEt = 24,
			fTime20TS = 25,

			//	for QIE 10
			fQIE10ADC256 = 26,
			fQIE10TDC64 = 27,

			nValueType = 28
		};

		std::string const vtitle[nValueType] = {
			"Entries", "Events", "1/Events", "Energy (GeV)", "Time (ns)",
			"ADC", "ADC", "ADC", "nom. fC", "nom. fC", "nom. fC", "Time Slice",
			"Time Slice", "LS", "Et", "Et", "Fine Grain Bit", "Ratio",
			"Digi Size", "Quantity", "Ratio", "Entries", "Entries", 
			"Summed dEt/Et", "Summed dEt", "Time Slice", "QIE10 ADC",
			"QIE10 TDC"
		};
		double const vmin[nValueType] = {
			0, 0, 0, 0, -50, 0, 0, 0, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5,
			-0.5, -0.5, -0.5, 0, -0.5, -10, 0, 0, 0, -0.05, 0, -0.5, -0.5, -0.5
		};
		double const vmax[nValueType] = {
			3000, 500, 1, 200, 50, 128, 5, 15, 10000, 1000, 3000, 9.5, 9.5, 
			4000.5, 255.5, 255.5, 1.5, 1.05, 10.5, 10, 2, 500, 100, 1.05, 2000,
			9.5, 55.5, 63.5
		};
		int const vnbins[nValueType] = {
			500, 100, 100, 400, 200, 128, 100, 300, 1000, 200, 600, 10, 
			200, 4000, 256, 128, 2, 200, 11, 20, 200, 500, 100, 200, 1000,10,
			256, 64
		};

		class ValueAxis : public Axis
		{
			public:
				friend class hcaldqm::Container;
				friend class hcaldqm::Container1D;
				friend class hcaldqm::Container2D;
				friend class hcaldqm::ContainerProf1D;
				friend class hcaldqm::ContainerProf2D;
				friend class hcaldqm::ContainerSingle2D;
				friend class hcaldqm::ContainerSingle1D;
				friend class hcaldqm::ContainerSingleProf1D;

			public:
				ValueAxis();
				ValueAxis(AxisType type, ValueType vtype, bool log=false);
				ValueAxis(AxisType type, ValueType vtype, int n, 
					double min,	double max, std::string title, bool log=false);
				virtual ~ValueAxis() {}
				virtual ValueAxis* makeCopy()
				{return new ValueAxis(_type, _vtype, _log);}

				virtual int getBin(int);
				virtual int getBin(double);

				virtual void setBitAxisLS(TObject *o)
				{
					if (_vtype==fLS)
						o->SetBit(BIT(BIT_OFFSET+BIT_AXIS_LS));
				}

			protected:
				virtual void _setup();

				ValueType _vtype;
		};
	}
}

#endif



