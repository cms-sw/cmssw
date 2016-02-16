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
	using namespace axis;
	class ContainerSingleProf1D : public ContainerSingle1D
	{
		public:
			ContainerSingleProf1D();
			ContainerSingleProf1D(std::string const& folder, 
				std::string const& nametitle, 
				axis::Axis *xaxis,
				axis::Axis *yaxis = new ValueAxis(fYaxis, axis::fEntries));
			virtual ~ContainerSingleProf1D() {}

			virtual void initialize(std::string const& folder, 
				std::string const& nametitle, 
				axis::Axis *xaxis,
				axis::Axis *yaxis = new ValueAxis(fYaxis, axis::fEntries),
				int debug=0);

			//	booking
			virtual void book(DQMStore::IBooker&,
				std::string subsystem="Hcal", std::string aux="");
	};
}

#endif
