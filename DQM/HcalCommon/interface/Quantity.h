#ifndef Quantity_h
#define Quantity_h

/**
 *	file:		Quantity.h
 *	Author:		Viktor Khristenko
 */

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"
#include "DQM/HcalCommon/interface/Constants.h"
#include "DQM/HcalCommon/interface/Utilities.h"

namespace hcaldqm
{
	namespace quantity
	{
		enum QuantityType
		{
			fDetectorQuantity = 0,
			fElectronicsQuantity = 1,
			fTrigTowerQuantity = 2,
			fValueQuantity = 3,
			fFlagQuantity = 4,
			fNone = 5,
			nQuantityType = 6
		};

		enum AxisType
		{
			fXAxis = 0,
			fYAxis = 1,
			fZAxis = 2,
			nAxisType = 3
		};

		class Quantity
		{
			public:
				Quantity() : _name("Quantity"), _isLog(false)
				{}
				Quantity(std::string const& name, bool isLog) : 
					_name(name), _isLog(isLog)
				{}
				virtual ~Quantity() {}

				virtual QuantityType	type() {return fNone;}
				virtual std::string		name() {return _name;}
				virtual bool			isLog() {return _isLog;}
				virtual	void			setAxisType(AxisType at) {_axistype=at;}
				virtual Quantity* makeCopy() 
				{return new Quantity(_name,_isLog);}

				virtual uint32_t getBin(HcalDetId const&) {return 1;}
				virtual uint32_t getBin(HcalElectronicsId const&) {return 1;}
				virtual uint32_t getBin(HcalTrigTowerDetId const&) {return 1;}
				virtual uint32_t getBin(int) {return 1;}
				virtual uint32_t getBin(double) {return 1;}

				virtual int getValue(HcalDetId const&) {return 0;}
				virtual int getValue(HcalElectronicsId const&) {return 0;}
				virtual int getValue(HcalTrigTowerDetId const&) {return 0;}
				virtual int getValue(int x) {return x;}
				virtual double getValue(double x) {return x;}

				virtual void setBits(TH1* o)
				{setLog(o);}
				virtual void setLog(TH1* o) 
				{
					if (_isLog)
						o->SetBit(BIT(BIT_OFFSET+_axistype));
				}

				virtual int nbins() {return 1;}
				virtual int wofnbins() {return nbins()+2;}
				virtual double min() {return 0;}
				virtual double max() {return 1;}
				virtual bool isCoordinate() {return false;}
				virtual std::vector<std::string> getLabels() 
				{return std::vector<std::string>();}

				virtual void setMax(double) {}
				virtual void setMin(double) {}
				virtual void setNbins(int) {}

			protected:
				std::string		_name;
				bool			_isLog;
				AxisType		_axistype;
		};
	}
}

#endif
