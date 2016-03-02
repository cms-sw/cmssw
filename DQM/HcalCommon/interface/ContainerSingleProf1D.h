#ifndef ContainerSingleProf1D_h
#define ContainerSingleProf1D_h

/*
 *	file:			ContainerSignle1D.h
 *	Author:			Viktor Khristenko
 *
 *	Description:
 *		Container to hold a single ME - for convenience of initialization
 */

#include "DQM/HcalCommon/interface/ContainerSingle1D.h"

#include <string>

namespace hcaldqm
{
	class ContainerSingleProf1D : public ContainerSingle1D
	{
		public:
			ContainerSingleProf1D();
			ContainerSingleProf1D(std::string const& folder, 
				Quantity*,
				Quantity *qy = new ValueQuantity(quantity::fN));
			virtual ~ContainerSingleProf1D() {}

			virtual void initialize(std::string const& folder, 
				Quantity*,
				Quantity *qy = new ValueQuantity(quantity::fN),
				int debug=0);
			virtual void initialize(std::string const& folder, 
				std::string const&,
				Quantity*,
				Quantity *qy = new ValueQuantity(quantity::fN),
				int debug=0);

			//	booking
			virtual void book(DQMStore::IBooker&,
				std::string subsystem="Hcal", std::string aux="");
			virtual void book(DQMStore*,
				std::string subsystem="Hcal", std::string aux="");
	};
}

#endif
