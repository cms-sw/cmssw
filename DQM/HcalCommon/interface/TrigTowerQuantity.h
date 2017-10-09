#ifndef TrigTowerQuantity_h
#define TrigTowerQuantity_h

/**
 *	file:		TrigTowerQuantity.h
 *	Author:		Viktor Khristenko
 */

#include "DQM/HcalCommon/interface/Quantity.h"

namespace hcaldqm
{
	namespace quantity
	{
		enum TrigTowerQuantityType
		{
			fTTiphi = 0,
			fTTieta = 1,
			fTTdepth = 2,
			fTTSubdet = 3,
			fTTSubdetPM = 4,
			fTTieta2x3 = 5,
			nTrigTowerQuantityType = 6
		};

		int getValue_TTiphi(HcalTrigTowerDetId const&);
		int getValue_TTieta(HcalTrigTowerDetId const&);
		int getValue_TTdepth(HcalTrigTowerDetId const&);
		int getValue_TTSubdet(HcalTrigTowerDetId const&);
		int getValue_TTSubdetPM(HcalTrigTowerDetId const&);
		int getValue_TTieta2x3(HcalTrigTowerDetId const&);
		uint32_t getBin_TTiphi(HcalTrigTowerDetId const&);
		uint32_t getBin_TTieta(HcalTrigTowerDetId const&);
		uint32_t getBin_TTdepth(HcalTrigTowerDetId const&);
		uint32_t getBin_TTSubdet(HcalTrigTowerDetId const&);
		uint32_t getBin_TTSubdetPM(HcalTrigTowerDetId const&);
		uint32_t getBin_TTieta2x3(HcalTrigTowerDetId const&);
		HcalTrigTowerDetId getTid_TTiphi(int);
		HcalTrigTowerDetId getTid_TTieta(int);
		HcalTrigTowerDetId getTid_TTdepth(int);
		HcalTrigTowerDetId getTid_TTSubdet(int);
		HcalTrigTowerDetId getTid_TTSubdetPM(int);
		HcalTrigTowerDetId getTid_TTieta2x3(int);
		std::vector<std::string> getLabels_TTiphi();
		std::vector<std::string> getLabels_TTieta();
		std::vector<std::string> getLabels_TTdepth();
		std::vector<std::string> getLabels_TTSubdet();
		std::vector<std::string> getLabels_TTSubdetPM();
		std::vector<std::string> getLabels_TTieta2x3();

		typedef int (*getValueType_tid)(HcalTrigTowerDetId const&);
		typedef uint32_t (*getBinType_tid)(HcalTrigTowerDetId const&);
		typedef HcalTrigTowerDetId (*getTid_tid)(int);
		typedef std::vector<std::string> (*getLabels_tid)();
		getValueType_tid const getValue_functions_tid[nTrigTowerQuantityType]= 
		{
			getValue_TTiphi, getValue_TTieta, getValue_TTdepth,
			getValue_TTSubdet, getValue_TTSubdetPM, getValue_TTieta2x3
		};
		getBinType_tid const getBin_functions_tid[nTrigTowerQuantityType] = {
			getBin_TTiphi, getBin_TTieta, getBin_TTdepth,
			getBin_TTSubdet, getBin_TTSubdetPM, getBin_TTieta2x3
		};
		getTid_tid const getTid_functions_tid[nTrigTowerQuantityType] = {
			getTid_TTiphi, getTid_TTieta, getTid_TTdepth,
			getTid_TTSubdet, getTid_TTSubdetPM, getTid_TTieta2x3
		};
		getLabels_tid const getLabels_functions_tid[nTrigTowerQuantityType] = {
			getLabels_TTiphi, getLabels_TTieta, getLabels_TTdepth,
			getLabels_TTSubdet, getLabels_TTSubdetPM, getLabels_TTieta2x3
		};
		std::string const name_tid[nTrigTowerQuantityType] = {
			"TTiphi", "TTieta", "TTdepth", "TTSubdet", "TTSubdetPM", "TTieta"
		};
		double const min_tid[nTrigTowerQuantityType] = {
			0.5, 0, -0.5, 0, 0, 0
		};
		double const max_tid[nTrigTowerQuantityType] = {
			72.5, 82, 0.5, 2, 4, 8
		};
		int const nbins_tid[nTrigTowerQuantityType] = {
			72, 82, 1, 2, 4, 8
		};

		class TrigTowerQuantity : public Quantity
		{
			public:
				TrigTowerQuantity() {}
				TrigTowerQuantity(TrigTowerQuantityType type, bool isLog=false):
					Quantity(name_tid[type], isLog), _type(type)
				{}
				virtual ~TrigTowerQuantity() {}
				virtual TrigTowerQuantity* makeCopy()
				{return new TrigTowerQuantity(_type, _isLog);}

				virtual int getValue(HcalTrigTowerDetId const& tid)
				{return getValue_functions_tid[_type](tid);}
				virtual uint32_t getBin(HcalTrigTowerDetId const& tid)
				{return getBin_functions_tid[_type](tid);}

				virtual QuantityType type() {return fTrigTowerQuantity;}
				virtual int nbins() {return nbins_tid[_type];}
				virtual double min() {return min_tid[_type];}
				virtual double max() {return max_tid[_type];}
				virtual bool isCoordinate() {return true;}
				virtual std::vector<std::string> getLabels() 
				{return getLabels_functions_tid[_type]();}

			protected:
				TrigTowerQuantityType _type;
		};
	}
}

#endif
