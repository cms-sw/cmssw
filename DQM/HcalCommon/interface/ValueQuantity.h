#ifndef ValueQuantity_h
#define ValueQuantity_h

#include "DQM/HcalCommon/interface/Quantity.h"
#include "DQM/HcalCommon/interface/Flag.h"

namespace hcaldqm
{
	namespace quantity
	{
		enum ValueQuantityType
		{
			fN = 0,
			fEvents = 1,
			fEnergy = 2,
			fTiming_ns = 3,
			fADC_128 = 4,
			fADC_5 = 5,
			fADC_15 = 6,
			ffC_10000 = 7,
			ffC_1000 = 8,
			ffC_3000 = 9,
			fTiming_TS = 10,
			fTiming_TS200 = 11,
			fLS = 12,
			fEt_256 = 13,
			fEt_128 = 14,
			fFG = 15,
			fRatio = 16,
			fDigiSize = 17,
			fAroundZero = 18,
			fRatio2 = 19,
			fdEtRatio = 20,
			fSumdEt = 21,
			fTiming_20TS = 22,

			fQIE10ADC_256 = 23,
			fQIE10TDC_64 = 24,
			fQIE10TDC_16 = 25,
			fDiffAbs = 26,

			fRatio_0to2 = 27,
			fN_to3000 = 28,
			fEnergyTotal = 29,
			fN_m0to10000 = 30,
			fEtCorr_256 = 31,
			fADCCorr_128 = 32,

			fBX = 33,
			fEnergy_1TeV = 34,
			fState = 35,
			nValueQuantityType = 36
		};
		std::string const name_value[nValueQuantityType] = {
			"N", "Events", "Energy", "Timing", "ADC", "ADC", "ADC",
			"fC", "fC", "fC", "Timing", "Timing", "LS", "Et", "Et",
			"FG", "Ratio", "DigiSize", "Q", "Ratio",
			"dEtRatio", "SumdEt", "Timing", "ADC", "TDC", "TDC",
			"Q", "Ratio", "N", "Energy", "N", "Et", "ADC", "BX",
			"Energy", "State"
		};
		double const min_value[nValueQuantityType] = {
			-0.05, 0, 0, -50, -0.5, -0.5, -0.5, 0, 0, 0, -0.5, 0, 0.5, 0,
			0, 0, 0, -0.5, -1, 0.5, 0, 0, -0.5, -0.5, -0.5, -0.5,
			0, 0, 0, 0, -0.05, -2, -2, -0.5, 0, flag::fNA
		};
		double const max_value[nValueQuantityType] = {
			1000, 1000, 200, 50, 127.5, 5, 15, 10000, 1000, 3000,
			9.5, 9.5, 4000.5, 255.5, 255.5, 2, 1, 20.5, 1, 1.5, 
			1, 1000, 9.5, 255.5, 63.5, 15.5, 1, 2, 3000, 100000, 10000,
			256, 128, 3600.5, 1000, flag::nState
		};
		int const nbins_value[nValueQuantityType] = {
			200, 200, 100, 200, 128, 100, 300, 1000, 200, 600, 
			10, 200, 4000, 256, 128, 2, 100, 20, 100, 100, 100, 100, 10,
			256, 64, 16, 200, 100, 3000, 500, 100, 258, 130, 3601, 200,
			flag::nState
		};

		class ValueQuantity : public Quantity
		{
			public:
				ValueQuantity() : _type(){}
				ValueQuantity(ValueQuantityType type, bool isLog=false) :
					Quantity(name_value[type], isLog), _type(type)
				{}
				virtual ~ValueQuantity() {}

				virtual ValueQuantity* makeCopy()
				{return new ValueQuantity(_type, _isLog);}

				//	get Value to be overriden
				virtual int getValue(int x)
				{return x;}
				virtual double getValue(double x)
				{return x;}

				//	standard properties
				virtual QuantityType type() {return fValueQuantity;}
				virtual int nbins() {return nbins_value[_type];}
				virtual double min() {return min_value[_type];}
				virtual double max() {return max_value[_type];}

				virtual void setBits(TH1* o)
				{Quantity::setBits(o);setLS(o);}
				virtual void setLS(TH1* o)
				{
					if (_type==fLS)
					{
						//	for LS axis - set the bit
						//	set extendable axes.
						o->SetBit(BIT(BIT_OFFSET+BIT_AXIS_LS));
		//				o->SetCanExtend(TH1::kXaxis);
					}
				}

			protected:
				ValueQuantityType _type;
		};

		class FlagQuantity : public ValueQuantity
		{
			public:
				FlagQuantity(){}
				FlagQuantity(std::vector<flag::Flag> const& flags) :
					_flags(flags) {}
				virtual ~FlagQuantity() {}
				
				virtual FlagQuantity* makeCopy()
				{return new FlagQuantity(_flags);}

				virtual std::string name() {return "Flag";}
				virtual int nbins() {return _flags.size();}
				virtual double min() {return 0;}
				virtual double max() {return _flags.size();}
				virtual int getValue(int f) {return f;}
				virtual uint32_t getBin(int f) {return f+1;}
				virtual std::vector<std::string> getLabels()
				{
					std::vector<std::string> vnames;
					for (std::vector<flag::Flag>::const_iterator
						it=_flags.begin(); it!=_flags.end(); ++it)
						vnames.push_back(it->_name);

					return vnames;
				}
			protected:

				std::vector<flag::Flag> _flags;
		};

		class LumiSection : public ValueQuantity
		{
			public:
				LumiSection() : ValueQuantity(fLS), _n(4000)
				{}
				LumiSection(int n) : ValueQuantity(fLS), 
					_n(n) 
				{}
				virtual ~LumiSection() {}
				
				virtual LumiSection* makeCopy()
				{return new LumiSection(_n);}

				virtual std::string name() {return "LS";}
				virtual int nbins() {return _n;}
				virtual double min() {return 1;}
				virtual double max() {return _n+1;}
				virtual int getValue(int l) {return l;}
				virtual uint32_t getBin(int l) 
				{return getValue(l);}
				virtual void setMax(double x) {_n=x;}

			protected:
				int _n;
		};

		class RunNumber : public ValueQuantity
		{
			public:
				RunNumber() {}
				RunNumber(std::vector<int> runs) :
					_runs(runs) 
				{}
				virtual ~RunNumber() {}

				virtual std::string name() {return "Run";}
				virtual int nbins() {return _runs.size();}
				virtual double min() {return 0;}
				virtual double max() {return _runs.size();}
				virtual std::vector<std::string> getLabels()
				{
					char name[10];
					std::vector<std::string> labels;
					for (uint32_t i=0; i<_runs.size(); i++)
					{
						sprintf(name, "%d", _runs[i]);
						labels.push_back(name);
					}
					return labels;
				}
				virtual int getValue(int run)
				{
					int ir = -1;
					for (uint32_t i=0; i<_runs.size(); i++)
						if (_runs[i]==run)
						{
							ir = (int)i;
							break;
						}

					if (ir==-1)
						throw cms::Exception("HCALDQM")
							<< "run number doens't exist " << run; 

					return ir;
				}

				virtual uint32_t getBin(int run)
				{
					return (this->getValue(run)+1);
				}

			protected:
				std::vector<int> _runs;
		};

		class EventNumber : public ValueQuantity
		{
			public:
				EventNumber() {}
				EventNumber(int nevents) :
					ValueQuantity(fN), _nevents(nevents)
				{}
				virtual ~EventNumber() {}

				virtual std::string name() {return "Event";}
				virtual int nbins() {return _nevents;}
				virtual double min() {return 0.5;}
				virtual double max() {return _nevents+0.5;}

			protected:
				int _nevents;
		};
	}
}

#endif
