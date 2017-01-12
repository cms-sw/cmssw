#ifndef ContainerSingle2D_h
#define ContainerSingle2D_h

/*
 *	file:			ContainerSignle2D.h
 *	Author:			Viktor Khristenko
 *
 *	Description:
 *		Container to hold a single ME - for convenience of initialization
 */

#include "DQM/HcalCommon/interface/Container.h"
#include "DQM/HcalCommon/interface/Quantity.h"
#include "DQM/HcalCommon/interface/ValueQuantity.h"

#include <string>

namespace hcaldqm
{
	using namespace quantity;
	class ContainerSingle2D : public Container
	{
		public:
			ContainerSingle2D();
			ContainerSingle2D(std::string const& folder, 
				Quantity*, Quantity*,
				Quantity *qz = new ValueQuantity(quantity::fN));
			ContainerSingle2D(std::string const& folder, 
				std::string const&,
				Quantity*, Quantity*,
				Quantity *qz = new ValueQuantity(quantity::fN),
				int debug=0);
			ContainerSingle2D(ContainerSingle2D const&);
			virtual ~ContainerSingle2D();

			virtual void initialize(std::string const& folder, 
				Quantity*, Quantity*,
				Quantity *qz = new ValueQuantity(quantity::fN),
				int debug=0);

			virtual void initialize(std::string const& folder, 
				std::string const&,
				Quantity*, Quantity*,
				Quantity *qz = new ValueQuantity(quantity::fN),
				int debug=0);

			//	booking
			virtual void book(DQMStore::IBooker&,
				std::string subsystem="Hcal", std::string aux="");
			virtual void book(DQMStore*,
				std::string subsystem="Hcal", std::string aux="");


			//	filling
			virtual void fill(int, int);
			virtual void fill(int, double);
			virtual void fill(int, double, double);
			virtual void fill(int, int, int);
			virtual void fill(int, int, double);
			virtual void fill(double, int);
			virtual void fill(double, double);
			virtual void fill(double, double, double);

			virtual double getBinContent(int, int);
			virtual double getBinContent(int, double);
			virtual double getBinContent(double, int);
			virtual double getBinContent(double, double);
			virtual double getBinEntries(int, int);
			virtual double getBinEntries(int, double);
			virtual double getBinEntries(double, int);
			virtual double getBinEntries(double, double);

			virtual void setBinContent(int, int, int);
			virtual void setBinContent(int, double, int);
			virtual void setBinContent(double, int, int);
			virtual void setBinContent(double, double, int);
			virtual void setBinContent(int, int, double);
			virtual void setBinContent(int, double, double);
			virtual void setBinContent(double, int, double);
			virtual void setBinContent(double, double, double);

			virtual void fill(HcalDetId const&);
			virtual void fill(HcalDetId const&, int);
			virtual void fill(HcalDetId const&, double);
			virtual void fill(HcalDetId const&, int, int);
			virtual void fill(HcalDetId const&, int, double);
			virtual void fill(HcalDetId const&, double, double);

			virtual double getBinContent(HcalDetId const&);
			virtual double getBinContent(HcalDetId const&, int);
			virtual double getBinContent(HcalDetId const&, double);
			virtual double getBinEntries(HcalDetId const&);
			virtual double getBinEntries(HcalDetId const&, int);
			virtual double getBinEntries(HcalDetId const&, double);

			virtual void setBinContent(HcalDetId const&, int);
			virtual void setBinContent(HcalDetId const&, double);
			virtual void setBinContent(HcalDetId const&, int, int);
			virtual void setBinContent(HcalDetId const&, int, double);
			virtual void setBinContent(HcalDetId const&, double, int);
			virtual void setBinContent(HcalDetId const&, double, double);

			virtual void fill(HcalElectronicsId const&);
			virtual void fill(HcalElectronicsId const&, int);
			virtual void fill(HcalElectronicsId const&, double);
			virtual void fill(HcalElectronicsId const&, int, int);
			virtual void fill(HcalElectronicsId const&, int, double);
			virtual void fill(HcalElectronicsId const&, double, double);

			virtual void fill(HcalDetId const&, HcalElectronicsId const&);
			virtual void fill(HcalDetId const&, HcalElectronicsId const&, 
				double);

			virtual double getBinContent(HcalElectronicsId const&);
			virtual double getBinContent(HcalElectronicsId const&, int);
			virtual double getBinContent(HcalElectronicsId const&, double);
			virtual double getBinEntries(HcalElectronicsId const&);
			virtual double getBinEntries(HcalElectronicsId const&, int);
			virtual double getBinEntries(HcalElectronicsId const&, double);

			virtual void setBinContent(HcalElectronicsId const&, int);
			virtual void setBinContent(HcalElectronicsId const&, double);
			virtual void setBinContent(HcalElectronicsId const&, int, int);
			virtual void setBinContent(HcalElectronicsId const&, int, double);
			virtual void setBinContent(HcalElectronicsId const&, double, int);
			virtual void setBinContent(HcalElectronicsId const&, double, double);

			virtual void fill(HcalTrigTowerDetId const&);
			virtual void fill(HcalTrigTowerDetId const&, int);
			virtual void fill(HcalTrigTowerDetId const&, double);
			virtual void fill(HcalTrigTowerDetId const&, int, int);
			virtual void fill(HcalTrigTowerDetId const&, int, double);
			virtual void fill(HcalTrigTowerDetId const&, double, double);

			virtual double getBinContent(HcalTrigTowerDetId const&);
			virtual double getBinContent(HcalTrigTowerDetId const&, int);
			virtual double getBinContent(HcalTrigTowerDetId const&, double);
			virtual double getBinEntries(HcalTrigTowerDetId const&);
			virtual double getBinEntries(HcalTrigTowerDetId const&, int);
			virtual double getBinEntries(HcalTrigTowerDetId const&, double);

			virtual void setBinContent(HcalTrigTowerDetId const&, int);
			virtual void setBinContent(HcalTrigTowerDetId const&, double);
			virtual void setBinContent(HcalTrigTowerDetId const&, int, int);
			virtual void setBinContent(HcalTrigTowerDetId const&, int, double);
			virtual void setBinContent(HcalTrigTowerDetId const&, double, int);
			virtual void setBinContent(HcalTrigTowerDetId const&, double, double);

			virtual void reset() {_me->Reset();}
			virtual void print() {std::cout << _qname << std::endl;}

			virtual void load(DQMStore::IGetter&, std::string subsystem="Hcal",
				std::string aux="");

			virtual void extendAxisRange(int);

		protected:
			MonitorElement					*_me;
			Quantity						*_qx;
			Quantity						*_qy;
			Quantity						*_qz;

			virtual void customize();
	};
}

#endif
