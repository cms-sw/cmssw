#ifndef ContainerSingleProf2D_h
#define ContainerSingleProf2D_h

/*
 *	file:			ContainerSignle2D.h
 *	Author:			Viktor Khristenko
 *
 *	Description:
 *		Container to hold a single ME - for convenience of initialization
 */

#include "DQM/HcalCommon/interface/ContainerSingle2D.h"

#include <string>

namespace hcaldqm
{
	class ContainerSingleProf2D : public ContainerSingle2D
	{
		public:
			ContainerSingleProf2D();
			ContainerSingleProf2D(std::string const& folder, 
				Quantity*, Quantity*,
				Quantity *qz = new ValueQuantity(quantity::fN));
			~ContainerSingleProf2D() override {}

			void initialize(std::string const& folder, 
				Quantity*, Quantity*,
				Quantity *qz = new ValueQuantity(quantity::fN),
				int debug=0) override;

			void initialize(std::string const& folder, 
				std::string const&,
				Quantity*, Quantity*,
				Quantity *qz = new ValueQuantity(quantity::fN),
				int debug=0) override;

			//	booking
			void book(DQMStore::IBooker&,
				std::string subsystem="Hcal", std::string aux="") override;
			void book(DQMStore*,
				std::string subsystem="Hcal", std::string aux="") override;

			void fill(int, int) override;
			void fill(int, double) override;
			void fill(int, double, double) override;
			void fill(int, int, int) override;
			void fill(int, int, double) override;
			void fill(double, int) override;
			void fill(double, double) override;
			void fill(double, double, double) override;

			virtual void fill(HcalDetId const&);
			virtual void fill(HcalDetId const&, int);
			virtual void fill(HcalDetId const&, double);
			virtual void fill(HcalDetId const&, int, int);
			virtual void fill(HcalDetId const&, int, double);
			virtual void fill(HcalDetId const&, double, double);

			virtual void fill(HcalElectronicsId const&);
			virtual void fill(HcalElectronicsId const&, int);
			virtual void fill(HcalElectronicsId const&, double);
			virtual void fill(HcalElectronicsId const&, int, int);
			virtual void fill(HcalElectronicsId const&, int, double);
			virtual void fill(HcalElectronicsId const&, double, double);

			virtual void fill(HcalDetId const&, HcalElectronicsId const&);
			virtual void fill(HcalDetId const&, HcalElectronicsId const&, 
				double);

			virtual void fill(HcalTrigTowerDetId const&);
			virtual void fill(HcalTrigTowerDetId const&, int);
			virtual void fill(HcalTrigTowerDetId const&, double);
			virtual void fill(HcalTrigTowerDetId const&, int, int);
			virtual void fill(HcalTrigTowerDetId const&, int, double);
			virtual void fill(HcalTrigTowerDetId const&, double, double);

	};
}

#endif
