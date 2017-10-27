#ifndef ValueQuantity_h
#define ValueQuantity_h

#include "DQM/HcalCommon/interface/Quantity.h"
#include "DQM/HcalCommon/interface/Flag.h"
#include "boost/unordered_map.hpp"
#include "boost/foreach.hpp"

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
			fN_to8000 = 28,
			fEnergyTotal = 29,
			fN_m0to10000 = 30,
			fEtCorr_256 = 31,
			fADCCorr_128 = 32,
			fBX = 33,
			fEnergy_1TeV = 34,
			fState = 35,
			fQIE10fC_400000 = 36,
			fDataSize = 37,
			fQIE10fC_2000 = 38,
			fQIE10fC_10000 = 39,
			fQIE8fC_1000_50 = 40,
			fTime_ns_250 = 41,
			fADC_256 = 42,
			ffC_generic_10000 = 43,
			ffC_generic_400000 = 44,
			fQIE10ADC_16 = 45,
			fDualAnodeAsymmetry = 46,
			fTimingRatio = 47,
			fQIE10fC_100000Coarse = 48,
			fBadTDC = 49,
		};
		const std::map<ValueQuantityType, std::string> name_value = {
			{fN,"N"},
			{fEvents,"Events"},
			{fEnergy,"Energy"},
			{fTiming_ns,"Timing"},
			{fADC_5,"ADC (QIE8)"},
			{fADC_15,"ADC (QIE8)"},
			{fADC_128,"ADC (QIE8)"},
			{fADC_256,"ADC"},
			{fQIE10ADC_256,"ADC (QIE10/11)"},
			{fQIE10ADC_16,"ADC (QIE10/11)"},
			{ffC_1000,"fC (QIE8)"},
			{ffC_3000,"fC (QIE8)"},
			{ffC_10000,"fC (QIE8)"},
			{fQIE8fC_1000_50,"fC (QIE8)"},
			{fQIE10fC_2000,"fC (QIE10/11)"},
			{fQIE10fC_10000,"fC (QIE10/11)"},
			{fQIE10fC_400000,"fC (QIE10/11)"},
			{ffC_generic_10000,"fC (QIE8/10/11)"},
			{ffC_generic_400000,"fC (QIE8/10/11)"},			
			{fTiming_TS,"Timing"},
			{fTiming_TS200,"Timing"},
			{fLS,"LS"},
			{fEt_256,"Et"},
			{fEt_128,"Et"},
			{fFG,"FG"},
			{fRatio,"Ratio"},
			{fDigiSize,"DigiSize"},
			{fAroundZero,"Q"},
			{fRatio2,"Ratio"},
			{fdEtRatio,"dEtRatio"},
			{fSumdEt,"SumdEt"},
			{fTiming_20TS,"Timing"},
			{fQIE10TDC_64,"TDC"},
			{fQIE10TDC_16,"TDC"},
			{fDiffAbs,"Q"},
			{fRatio_0to2,"Ratio"},
			{fN_to8000,"N"},
			{fEnergyTotal,"Energy"},
			{fN_m0to10000,"N"},
			{fEtCorr_256,"Et"},
			{fADCCorr_128,"ADC"},
			{fBX,"BX"},
			{fEnergy_1TeV,"Energy"},
			{fState,"State"},
			{fDataSize,"FED Data Size (kB)"},
			{fTime_ns_250,"Time (ns)"},
			{fDualAnodeAsymmetry, "(q_{1}-q_{2})/(q_{1}+q_{2})"},
			{fTimingRatio, "q_{SOI+1}/q_{SOI}"},
			{fQIE10fC_100000Coarse,"fC (QIE10/11)"},
			{fBadTDC, "TDC"},
		};
		const std::map<ValueQuantityType, double> min_value = {
			{fN,-0.05},
			{fEvents,0},
			{fEnergy,0},
			{fTiming_ns,-50},
			{fADC_128,-0.5},
			{fADC_5,-0.5},
			{fADC_15,-0.5},
			{ffC_10000,0},
			{ffC_1000,0},
			{ffC_3000,0},
			{fTiming_TS,-0.5},
			{fTiming_TS200,0},
			{fLS,0.5},
			{fEt_256,0},
			{fEt_128,0},
			{fFG,0},
			{fRatio,0},
			{fDigiSize,-0.5},
			{fAroundZero,-1},
			{fRatio2,0.5},
			{fdEtRatio,0},
			{fSumdEt,0},
			{fTiming_20TS,-0.5},
			{fQIE10ADC_256,-0.5},
			{fQIE10ADC_16,-0.5},
			{fQIE10TDC_64,-0.5},
			{fQIE10TDC_16,-0.5},
			{fDiffAbs,0},
			{fRatio_0to2,0},
			{fN_to8000,0},
			{fEnergyTotal,0},
			{fN_m0to10000,-0.05},
			{fEtCorr_256,-1},
			{fADCCorr_128,-2},
			{fBX,-0.5},
			{fEnergy_1TeV,0},
			{fState,flag::fNA},
			{fQIE10fC_400000,0},
			{fDataSize,0},
			{fQIE10fC_2000,0},
			{fQIE10fC_10000,0},
			{fQIE8fC_1000_50,0},
			{fTime_ns_250,-0.5},
			{fADC_256,-0.5},
			{ffC_generic_10000,0.},
			{ffC_generic_400000,0.},	
			{fDualAnodeAsymmetry,-1.},	
			{fTimingRatio,0.},	
			{fQIE10fC_100000Coarse,0},
			{fBadTDC,49.5},
		};
		const std::map<ValueQuantityType, double> max_value = {
			{fN,1000},
			{fEvents,1000},
			{fEnergy,200},
			{fTiming_ns,50},
			{fADC_128,127.5},
			{fADC_5,5},
			{fADC_15,15},
			{ffC_10000,10000},
			{ffC_1000,1000},
			{ffC_3000,3000},
			{fTiming_TS,9.5},
			{fTiming_TS200,9.5},
			{fLS,4000.5},
			{fEt_256,255.5},
			{fEt_128,255.5},
			{fFG,2},
			{fRatio,1},
			{fDigiSize,20.5},
			{fAroundZero,1},
			{fRatio2,1.5},
			{fdEtRatio,1},
			{fSumdEt,1000},
			{fTiming_20TS,9.5},
			{fQIE10ADC_256,255.5},
			{fQIE10ADC_16,15.5},
			{fQIE10TDC_64,63.5},
			{fQIE10TDC_16,15.5},
			{fDiffAbs,1},
			{fRatio_0to2,2},
			{fN_to8000,8000},
			{fEnergyTotal,100000},
			{fN_m0to10000,10000},
			{fEtCorr_256,257},
			{fADCCorr_128,128},
			{fBX,3600.5},
			{fEnergy_1TeV,1000},
			{fState,flag::nState},
			{fQIE10fC_400000,400000},
			{fDataSize,100},
			{fQIE10fC_2000,2000},
			{fQIE10fC_10000,10000},
			{fQIE8fC_1000_50,1000},
			{fTime_ns_250,249.5},
			{fADC_256,255.5},
			{ffC_generic_10000,10000},
			{ffC_generic_400000,400000},	
			{fDualAnodeAsymmetry,1.},
			{fTimingRatio,2.5},	
			{fQIE10fC_100000Coarse,100000},
			{fBadTDC,61.5},
		};
		const std::map<ValueQuantityType, int> nbins_value = {
			{fN,200},
			{fEvents,200},
			{fEnergy,100},
			{fTiming_ns,200},
			{fADC_128,128},
			{fADC_5,100},
			{fADC_15,300},
			{ffC_10000,1000},
			{ffC_1000,200},
			{ffC_3000,600},
			{fTiming_TS,10},
			{fTiming_TS200,200},
			{fLS,4000},
			{fEt_256,256},
			{fEt_128,128},
			{fFG,2},
			{fRatio,100},
			{fDigiSize,21},
			{fAroundZero,100},
			{fRatio2,100},
			{fdEtRatio,100},
			{fSumdEt,100},
			{fTiming_20TS,10},
			{fQIE10ADC_256,256},
			{fQIE10TDC_64,64},
			{fQIE10TDC_16,16},
			{fDiffAbs,200},
			{fRatio_0to2,100},
			{fN_to8000,8000},
			{fEnergyTotal,500},
			{fN_m0to10000,100},
			{fEtCorr_256,258},
			{fADCCorr_128,130},
			{fBX,3601},
			{fEnergy_1TeV,200},
			{fState,flag::nState},
			{fQIE10fC_400000,4000},
			{fDataSize,100},
			{fQIE10fC_2000,100},
			{fQIE10fC_10000,500},
			{fQIE8fC_1000_50,50},
			{fTime_ns_250,250},
			{fADC_256,256},
			{ffC_generic_10000,10000},
			{ffC_generic_400000,10000},	
			{fDualAnodeAsymmetry,40},
			{fTimingRatio,50},
			{fQIE10fC_100000Coarse,1000},
			{fBadTDC,12}
		};
		class ValueQuantity : public Quantity
		{
			public:
				ValueQuantity() : _type(){}
				ValueQuantity(ValueQuantityType type, bool isLog=false) :
					Quantity(name_value.at(type), isLog), _type(type)
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
				virtual int nbins() {return nbins_value.at(_type);}
				virtual double min() {return min_value.at(_type);}
				virtual double max() {return max_value.at(_type);}

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

		class EventType : public ValueQuantity
		{
			public:
				EventType() {}
				EventType(std::vector<uint32_t> const& vtypes):
					ValueQuantity(fN)
				{this->setup(vtypes);}
				virtual ~EventType() {}

				virtual void setup(std::vector<uint32_t> const& vtypes)
				{
					std::cout << "SIZE = " << vtypes.size() << std::endl;
					for (uint32_t i=0; i<vtypes.size(); i++)
						_types.insert(std::make_pair((uint32_t)vtypes[i], i));
				}
				virtual int getValue(int v)
				{
					return _types[(uint32_t)v];
				}
				virtual uint32_t getBin(int v)
				{
					return getValue(v)+1;
				}

				virtual int nbins() {return _types.size();}
				virtual double min() {return 0;}
				virtual double max() {return _types.size();}
				virtual std::string name() {return "Event Type";}

			protected:
				typedef boost::unordered_map<uint32_t, int> TypeMap;
				TypeMap _types;

			public:
				virtual std::vector<std::string> getLabels()
				{
					std::vector<std::string> labels(_types.size());
					std::cout << "SIZE = " << _types.size() << std::endl;
					BOOST_FOREACH(TypeMap::value_type &v, _types)
					{
						labels[v.second] = utilities::ogtype2string((OrbitGapType) 
							v.first);
					}
					return labels;
				}
				virtual EventType* makeCopy()
				{
					std::vector<uint32_t> vtypes;
					BOOST_FOREACH(TypeMap::value_type &p, _types)
					{
						vtypes.push_back(p.first);
					}

					std::sort(vtypes.begin(), vtypes.end());
					return new EventType(vtypes);
				}
		};
	}
}

#endif
