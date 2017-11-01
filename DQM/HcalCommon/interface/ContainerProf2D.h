#ifndef ContainerProf2D_h
#define ContainerProf2D_h

/*
 *	file:		ContainerProf2D.h
 *	Author:		Viktor Khristenko
 *
 *	Description:
 *		Container to hold TProfile or like
 *
 */

#include "DQM/HcalCommon/interface/Container2D.h"

#include <vector>
#include <string>

namespace hcaldqm
{
	using namespace quantity;
	class ContainerProf2D : public Container2D
	{
		public:
			ContainerProf2D();
			ContainerProf2D(std::string const& folder, 
				hashfunctions::HashType,
				Quantity*, Quantity*, 
				Quantity* qz = new ValueQuantity(quantity::fEnergy));
			~ContainerProf2D() override {}
			
			void initialize(std::string const& folder, 
				hashfunctions::HashType,
				Quantity*, Quantity*,
				Quantity *qz = new ValueQuantity(quantity::fEnergy),
				int debug=0) override;

			void initialize(std::string const& folder, 
				std::string const& qname,
				hashfunctions::HashType,
				Quantity*, Quantity*,
				Quantity *qz = new ValueQuantity(quantity::fEnergy),
				int debug=0) override;

			void book(DQMStore::IBooker&,
				HcalElectronicsMap const*,
				std::string subsystem="Hcal", std::string aux="") override;
			void book(DQMStore::IBooker&,
				HcalElectronicsMap const*, filter::HashFilter const&,
				std::string subsystem="Hcal", std::string aux="") override;
			void book(DQMStore*,
				HcalElectronicsMap const*,
				std::string subsystem="Hcal", std::string aux="") override;
			void book(DQMStore*,
				HcalElectronicsMap const*, filter::HashFilter const&,
				std::string subsystem="Hcal", std::string aux="") override;

		protected:
	};
}


#endif








