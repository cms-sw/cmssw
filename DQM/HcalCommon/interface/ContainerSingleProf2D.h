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
			virtual ~ContainerSingleProf2D() {}

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
	};
}

#endif
