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
	using namespace axis;
	class ContainerProf2D : public Container2D
	{
		public:
			ContainerProf2D();
			ContainerProf2D(std::string const& folder, 
				std::string const &nametitle,
				MapperType mt, 
				Axis *xaxis, 
				Axis *yaxis = new CoordinateAxis(axis::fYaxis, axis::fiphi), 
				Axis *zaxis = new ValueAxis(axis::fZaxis, axis::fEntries));
			virtual ~ContainerProf2D() {}
			
			virtual void initialize(std::string const& folder, 
				std::string const &nametitle,
				MapperType mt, 
				Axis *xaxis, 
				Axis *yaxis = new CoordinateAxis(axis::fYaxis, axis::fiphi), 
				Axis *zaxis = new ValueAxis(axis::fZaxis, axis::fEntries),
				int debug=0);

			virtual void book(DQMStore::IBooker&,
				std::string subsystem="Hcal", std::string aux="");

		protected:
	};
}


#endif








