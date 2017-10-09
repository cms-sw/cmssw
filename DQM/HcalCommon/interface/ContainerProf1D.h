#ifndef ContainerProf1D_h
#define ContainerProf1D_h

/*
 *	file:		ContainerProf1D.h
 *	Author:		Viktor Khristenko
 *
 *	Description:
 *		Container to hold TProfiles. 
 *		Direct Inheritance from Container1D + some more funcs
 *
 */

#include "DQM/HcalCommon/interface/Container1D.h"

#include <vector>
#include <string>

namespace hcaldqm
{
	using namespace quantity;
	using namespace mapper;
	class ContainerProf1D : public Container1D
	{
		public:
			ContainerProf1D();
			ContainerProf1D(std::string const& folder, 
				hashfunctions::HashType, 
				Quantity *, Quantity*);
			virtual ~ContainerProf1D() {}

			virtual void initialize(std::string const& folder, 
				hashfunctions::HashType, 
				Quantity*, Quantity*,
				int debug=0);

			virtual void initialize(std::string const& folder, 
				std::string const& qname,
				hashfunctions::HashType, 
				Quantity*, Quantity*,
				int debug=0);

			//	booking
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
	};
}


#endif








