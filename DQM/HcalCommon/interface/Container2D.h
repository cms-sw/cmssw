#ifndef Container2D_h
#define Container2D_h

/*
 *	file:		Container2D.h
 *	Author:		Viktor Khristenko
 *
 *	Description:
 *		Container to hold TH2D or like
 *
 */

#include "DQM/HcalCommon/interface/Container1D.h"

#include <vector>
#include <string>

namespace hcaldqm
{
	class Container2D : public Container1D
	{
		public:
			Container2D();
			Container2D(std::string const& folder,
				hashfunctions::HashType, Quantity*, Quantity*,
				Quantity *qz = new ValueQuantity(quantity::fN));
			virtual ~Container2D();

			//	Initialize Container
			//	@folder
			//	@nametitle, 
			virtual void initialize(std::string const& folder, 
					hashfunctions::HashType, Quantity*, Quantity*,
					Quantity *qz = new ValueQuantity(quantity::fN),
					int debug=0);
			
			//	@qname - quantity name replacer
			virtual void initialize(std::string const& folder, 
				std::string const& qname,
				hashfunctions::HashType, Quantity*, Quantity*,
				Quantity *qz = new ValueQuantity(quantity::fN),
				int debug=0);

			//	redeclare what to override
			virtual void fill(HcalDetId const&);
			virtual void fill(HcalDetId const&, int);
			virtual void fill(HcalDetId const&, double);
			virtual void fill(HcalDetId const&, int, double);
			virtual void fill(HcalDetId const&, int, int);
			virtual void fill(HcalDetId const&, double, double);

			virtual double getBinEntries(HcalDetId const&);
			virtual double getBinEntries(HcalDetId const&, int);
			virtual double getBinEntries(HcalDetId const&, double);
			virtual double getBinEntries(HcalDetId const&, int, int);
			virtual double getBinEntries(HcalDetId const&, int, double);
			virtual double getBinEntries(HcalDetId const&, double, double);

			virtual double getBinContent(HcalDetId const&);
			virtual double getBinContent(HcalDetId const&, int);
			virtual double getBinContent(HcalDetId const&, double);
			virtual double getBinContent(HcalDetId const&, int, int);
			virtual double getBinContent(HcalDetId const&, int, double);
			virtual double getBinContent(HcalDetId const&, double, double);

			virtual void setBinContent(HcalDetId const&, int);
			virtual void setBinContent(HcalDetId const&, double);
			virtual void setBinContent(HcalDetId const&, int, int);
			virtual void setBinContent(HcalDetId const&, int, double);
			virtual void setBinContent(HcalDetId const&, double, int);
			virtual void setBinContent(HcalDetId const&, double, double);
			virtual void setBinContent(HcalDetId const&, int, int, int);
			virtual void setBinContent(HcalDetId const&, int, double, int);
			virtual void setBinContent(HcalDetId const&, double, int, int);
			virtual void setBinContent(HcalDetId const&, double, double, int);
			virtual void setBinContent(HcalDetId const&, int, int, double);
			virtual void setBinContent(HcalDetId const&, int, double, double);
			virtual void setBinContent(HcalDetId const&, double, int, double);
			virtual void setBinContent(HcalDetId const&, double, double, 
				double);

			virtual void fill(HcalElectronicsId const&);
			virtual void fill(HcalElectronicsId const&, int);
			virtual void fill(HcalElectronicsId const&, double);
			virtual void fill(HcalElectronicsId const&, int, double);
			virtual void fill(HcalElectronicsId const&, int, int);
			virtual void fill(HcalElectronicsId const&, double, double);

			virtual double getBinEntries(HcalElectronicsId const&);
			virtual double getBinEntries(HcalElectronicsId const&, int);
			virtual double getBinEntries(HcalElectronicsId const&, double);
			virtual double getBinEntries(HcalElectronicsId const&, int, int);
			virtual double getBinEntries(HcalElectronicsId const&, int, double);
			virtual double getBinEntries(HcalElectronicsId const&, double, 
				double);

			virtual double getBinContent(HcalElectronicsId const&);
			virtual double getBinContent(HcalElectronicsId const&, int);
			virtual double getBinContent(HcalElectronicsId const&, double);
			virtual double getBinContent(HcalElectronicsId const&, int, int);
			virtual double getBinContent(HcalElectronicsId const&, int, double);
			virtual double getBinContent(HcalElectronicsId const&, double, 
				double);

			virtual void setBinContent(HcalElectronicsId const&, int);
			virtual void setBinContent(HcalElectronicsId const&, double);
			virtual void setBinContent(HcalElectronicsId const&, int, int);
			virtual void setBinContent(HcalElectronicsId const&, int, double);
			virtual void setBinContent(HcalElectronicsId const&, double, int);
			virtual void setBinContent(HcalElectronicsId const&, double, double);
			virtual void setBinContent(HcalElectronicsId const&, int, int, int);
			virtual void setBinContent(HcalElectronicsId const&, int, double, int);
			virtual void setBinContent(HcalElectronicsId const&, double, int, int);
			virtual void setBinContent(HcalElectronicsId const&, double, double, int);
			virtual void setBinContent(HcalElectronicsId const&, int, int, double);
			virtual void setBinContent(HcalElectronicsId const&, int, double, double);
			virtual void setBinContent(HcalElectronicsId const&, double, int, double);
			virtual void setBinContent(HcalElectronicsId const&, double, double, 
				double);

			virtual void fill(HcalTrigTowerDetId const&);
			virtual void fill(HcalTrigTowerDetId const&, int);
			virtual void fill(HcalTrigTowerDetId const&, double);
			virtual void fill(HcalTrigTowerDetId const&, int, int);
			virtual void fill(HcalTrigTowerDetId const&, int, double);
			virtual void fill(HcalTrigTowerDetId const&, double, double);

			virtual double getBinEntries(HcalTrigTowerDetId const&);
			virtual double getBinEntries(HcalTrigTowerDetId const&, int);
			virtual double getBinEntries(HcalTrigTowerDetId const&, double);
			virtual double getBinEntries(HcalTrigTowerDetId const&, int, int);
			virtual double getBinEntries(HcalTrigTowerDetId const&, int, 
				double);
			virtual double getBinEntries(HcalTrigTowerDetId const&, 
				double, double);

			virtual double getBinContent(HcalTrigTowerDetId const&);
			virtual double  getBinContent(HcalTrigTowerDetId const&, int);
			virtual double getBinContent(HcalTrigTowerDetId const&, double);
			virtual double getBinContent(HcalTrigTowerDetId const&, int, int);
			virtual double getBinContent(HcalTrigTowerDetId const&, int, double);
			virtual double getBinContent(HcalTrigTowerDetId const&, 
				double, double);

			virtual void setBinContent(HcalTrigTowerDetId const&, int);
			virtual void setBinContent(HcalTrigTowerDetId const&, double);
			virtual void setBinContent(HcalTrigTowerDetId const&, int, int);
			virtual void setBinContent(HcalTrigTowerDetId const&, int, double);
			virtual void setBinContent(HcalTrigTowerDetId const&, double, int);
			virtual void setBinContent(HcalTrigTowerDetId const&, double, double);
			virtual void setBinContent(HcalTrigTowerDetId const&, int, int, int);
			virtual void setBinContent(HcalTrigTowerDetId const&, int, double, int);
			virtual void setBinContent(HcalTrigTowerDetId const&, double, int, int);
			virtual void setBinContent(HcalTrigTowerDetId const&, double, double, int);
			virtual void setBinContent(HcalTrigTowerDetId const&, int, int, double);
			virtual void setBinContent(HcalTrigTowerDetId const&, int, double, double);
			virtual void setBinContent(HcalTrigTowerDetId const&, double, int, double);
			virtual void setBinContent(HcalTrigTowerDetId const&, double, double, 
				double);

			//	booking. see Container1D.h
			virtual void book(DQMStore::IBooker&,
				HcalElectronicsMap const*,
				std::string subsystem="Hcal", std::string aux="");
			virtual void book(DQMStore::IBooker&,
				HcalElectronicsMap const*, filter::HashFilter const&,
				std::string subsystem="Hcal", std::string aux="");
			virtual void book(DQMStore*,
				HcalElectronicsMap const*,
				std::string subsystem="Hcal", std::string aux="");
			virtual void book(DQMStore*,
				HcalElectronicsMap const*, filter::HashFilter const&,
				std::string subsystem="Hcal", std::string aux="");

		protected:
			Quantity	*_qz;

			virtual void customize(MonitorElement*);
	};
}


#endif








