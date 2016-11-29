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
			virtual void fill(HcalDetId const&) override ;
			virtual void fill(HcalDetId const&, int) override;
			virtual void fill(HcalDetId const&, double) override;
			virtual void fill(HcalDetId const&, int, double) override;
			virtual void fill(HcalDetId const&, int, int) override;
			virtual void fill(HcalDetId const&, double, double) override;

			virtual double getBinEntries(HcalDetId const&) override;
			virtual double getBinEntries(HcalDetId const&, int) override;
			virtual double getBinEntries(HcalDetId const&, double) override;
			virtual double getBinEntries(HcalDetId const&, int, int) override;
			virtual double getBinEntries(HcalDetId const&, int, double) override;
			virtual double getBinEntries(HcalDetId const&, double, double) override;

			virtual double getBinContent(HcalDetId const&) override;
			virtual double getBinContent(HcalDetId const&, int) override;
			virtual double getBinContent(HcalDetId const&, double) override;
			virtual double getBinContent(HcalDetId const&, int, int) override;
			virtual double getBinContent(HcalDetId const&, int, double) override;
			virtual double getBinContent(HcalDetId const&, double, double) override;

			virtual void setBinContent(HcalDetId const&, int) override;
			virtual void setBinContent(HcalDetId const&, double) override;
			virtual void setBinContent(HcalDetId const&, int, int) override;
			virtual void setBinContent(HcalDetId const&, int, double) override;
			virtual void setBinContent(HcalDetId const&, double, int) override;
			virtual void setBinContent(HcalDetId const&, double, double) override;
			virtual void setBinContent(HcalDetId const&, int, int, int) override;
			virtual void setBinContent(HcalDetId const&, int, double, int) override;
			virtual void setBinContent(HcalDetId const&, double, int, int) override;
			virtual void setBinContent(HcalDetId const&, double, double, int) override;
			virtual void setBinContent(HcalDetId const&, int, int, double) override;
			virtual void setBinContent(HcalDetId const&, int, double, double) override;
			virtual void setBinContent(HcalDetId const&, double, int, double) override;
			virtual void setBinContent(HcalDetId const&, double, double, 
				double) override;

			virtual void fill(HcalElectronicsId const&) override;
			virtual void fill(HcalElectronicsId const&, int) override;
			virtual void fill(HcalElectronicsId const&, double) override;
			virtual void fill(HcalElectronicsId const&, int, double) override;
			virtual void fill(HcalElectronicsId const&, int, int) override;
			virtual void fill(HcalElectronicsId const&, double, double) override;

			virtual double getBinEntries(HcalElectronicsId const&) override;
			virtual double getBinEntries(HcalElectronicsId const&, int) override;
			virtual double getBinEntries(HcalElectronicsId const&, double) override;
			virtual double getBinEntries(HcalElectronicsId const&, int, int) override;
			virtual double getBinEntries(HcalElectronicsId const&, int, double) override;
			virtual double getBinEntries(HcalElectronicsId const&, double, 
				double) override;

			virtual double getBinContent(HcalElectronicsId const&) override;
			virtual double getBinContent(HcalElectronicsId const&, int) override;
			virtual double getBinContent(HcalElectronicsId const&, double) override;
			virtual double getBinContent(HcalElectronicsId const&, int, int) override;
			virtual double getBinContent(HcalElectronicsId const&, int, double) override;
			virtual double getBinContent(HcalElectronicsId const&, double, 
				double) override;

			virtual void setBinContent(HcalElectronicsId const&, int) override;
			virtual void setBinContent(HcalElectronicsId const&, double) override;
			virtual void setBinContent(HcalElectronicsId const&, int, int) override;
			virtual void setBinContent(HcalElectronicsId const&, int, double) override;
			virtual void setBinContent(HcalElectronicsId const&, double, int) override;
			virtual void setBinContent(HcalElectronicsId const&, double, double) override;
			virtual void setBinContent(HcalElectronicsId const&, int, int, int) override;
			virtual void setBinContent(HcalElectronicsId const&, int, double, int) override;
			virtual void setBinContent(HcalElectronicsId const&, double, int, int) override;
			virtual void setBinContent(HcalElectronicsId const&, double, double, int) override;
			virtual void setBinContent(HcalElectronicsId const&, int, int, double) override;
			virtual void setBinContent(HcalElectronicsId const&, int, double, double) override;
			virtual void setBinContent(HcalElectronicsId const&, double, int, double) override;
			virtual void setBinContent(HcalElectronicsId const&, double, double, 
				double) override;

			virtual void fill(HcalTrigTowerDetId const&) override;
			virtual void fill(HcalTrigTowerDetId const&, int) override;
			virtual void fill(HcalTrigTowerDetId const&, double) override;
			virtual void fill(HcalTrigTowerDetId const&, int, int) override;
			virtual void fill(HcalTrigTowerDetId const&, int, double) override;
			virtual void fill(HcalTrigTowerDetId const&, double, double) override;

			virtual double getBinEntries(HcalTrigTowerDetId const&) override;
			virtual double getBinEntries(HcalTrigTowerDetId const&, int) override;
			virtual double getBinEntries(HcalTrigTowerDetId const&, double) override;
			virtual double getBinEntries(HcalTrigTowerDetId const&, int, int) override;
			virtual double getBinEntries(HcalTrigTowerDetId const&, int, 
				double) override;
			virtual double getBinEntries(HcalTrigTowerDetId const&, 
				double, double) override;

			virtual double getBinContent(HcalTrigTowerDetId const&) override;
			virtual double  getBinContent(HcalTrigTowerDetId const&, int) override;
			virtual double getBinContent(HcalTrigTowerDetId const&, double) override;
			virtual double getBinContent(HcalTrigTowerDetId const&, int, int) override;
			virtual double getBinContent(HcalTrigTowerDetId const&, int, double) override;
			virtual double getBinContent(HcalTrigTowerDetId const&, 
				double, double) override;

			virtual void setBinContent(HcalTrigTowerDetId const&, int) override;
			virtual void setBinContent(HcalTrigTowerDetId const&, double) override;
			virtual void setBinContent(HcalTrigTowerDetId const&, int, int) override;
			virtual void setBinContent(HcalTrigTowerDetId const&, int, double) override;
			virtual void setBinContent(HcalTrigTowerDetId const&, double, int) override;
			virtual void setBinContent(HcalTrigTowerDetId const&, double, double) override;
			virtual void setBinContent(HcalTrigTowerDetId const&, int, int, int) override;
			virtual void setBinContent(HcalTrigTowerDetId const&, int, double, int) override;
			virtual void setBinContent(HcalTrigTowerDetId const&, double, int, int) override;
			virtual void setBinContent(HcalTrigTowerDetId const&, double, double, int) override;
			virtual void setBinContent(HcalTrigTowerDetId const&, int, int, double) override;
			virtual void setBinContent(HcalTrigTowerDetId const&, int, double, double) override;
			virtual void setBinContent(HcalTrigTowerDetId const&, double, int, double) override;
			virtual void setBinContent(HcalTrigTowerDetId const&, double, double, 
				double) override;

			//	booking. see Container1D.h
			virtual void book(DQMStore::IBooker&,
				HcalElectronicsMap const*,
				std::string subsystem="Hcal", std::string aux="") override;
			virtual void book(DQMStore::IBooker&,
				HcalElectronicsMap const*, filter::HashFilter const&,
				std::string subsystem="Hcal", std::string aux="") override;
			virtual void book(DQMStore*,
				HcalElectronicsMap const*,
				std::string subsystem="Hcal", std::string aux="") override;
			virtual void book(DQMStore*,
				HcalElectronicsMap const*, filter::HashFilter const&,
				std::string subsystem="Hcal", std::string aux="") override;

		protected:
			Quantity	*_qz;

			virtual void customize(MonitorElement*) override;
	};
}


#endif








