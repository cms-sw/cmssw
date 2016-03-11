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
	class ContainerProf1D : public Container1D
	{
		public:
			ContainerProf1D();
			ContainerProf1D(std::string const& folder, 
				std::string const& nametitle, mapper::MapperType mt, 
				axis::Axis *xaxis, 
				axis::Axis *yaxis = new axis::ValueAxis(axis::fYaxis, 
				axis::fEntries));
			virtual ~ContainerProf1D() {}

			virtual void initialize(std::string const& folder, 
				std::string const& nametitle, mapper::MapperType mt, 
				axis::Axis *xaxis, 
				axis::Axis *yaxis = new axis::ValueAxis(axis::fYaxis, 
				axis::fEntries), int debug=0);

			//	booking
			virtual void book(DQMStore::IBooker&,
				std::string subsystem="Hcal", std::string aux="");

		protected:
	};
}


#endif








