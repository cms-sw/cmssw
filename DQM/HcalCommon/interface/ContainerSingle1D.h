#ifndef ContainerSingle1D_h
#define ContainerSingle1D_h

/*
 *	file:			ContainerSignle1D.h
 *	Author:			Viktor Khristenko
 *
 *	Description:
 *		Container to hold a single ME - for convenience of initialization
 */

#include "DQM/HcalCommon/interface/Container.h"
#include "DQM/HcalCommon/interface/ValueAxis.h"
#include "DQM/HcalCommon/interface/CoordinateAxis.h"

#include <string>

namespace hcaldqm
{
	using namespace axis;
	class ContainerSingle1D : public Container
	{
		public:
			ContainerSingle1D();
			ContainerSingle1D(std::string const& folder, 
				std::string const& nametitle, 
				axis::Axis *xaxis,
				axis::Axis *yaxis = new ValueAxis(fYaxis, axis::fEntries));
			ContainerSingle1D(ContainerSingle1D const&);
			virtual ~ContainerSingle1D();
			
			virtual void initialize(std::string const& folder, 
				std::string const& nametitle, 
				axis::Axis *xaxis,
				axis::Axis *yaxis = new ValueAxis(fYaxis, axis::fEntries),
				int debug=0);

			//	booking
			virtual void book(DQMStore::IBooker&,
				std::string subsystem="Hcal", std::string aux="");

			//	filling
			virtual void fill(int);
			virtual void fill(double);
			virtual void fill(int, int);
			virtual void fill(int, double);
			virtual void fill(double, int);
			virtual void fill(double, double);

			virtual void fill(HcalDetId const&);
			virtual void fill(HcalDetId const&, double);
			virtual void fill(HcalDetId const&, double, double);

			virtual void fill(HcalElectronicsId const&);
			virtual void fill(HcalElectronicsId const&, double);

		protected:
			MonitorElement				*_me;
			Axis						*_xaxis;
			Axis						*_yaxis;
	};
}

#endif
