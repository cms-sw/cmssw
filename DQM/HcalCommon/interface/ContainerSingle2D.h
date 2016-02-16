#ifndef ContainerSingle2D_h
#define ContainerSingle2D_h

/*
 *	file:			ContainerSignle2D.h
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
	class ContainerSingle2D : public Container
	{
		public:
			ContainerSingle2D();
			ContainerSingle2D(std::string const& folder, 
				std::string const& nametitle, 
				axis::Axis *xaxis,
				axis::Axis *yaxis = new CoordinateAxis(fYaxis, axis::fiphi), 
				axis::Axis *zaxis = new ValueAxis(fZaxis, fEntries));
			virtual ~ContainerSingle2D();

			virtual void initialize(std::string const& folder, 
				std::string const& nametitle, 
				axis::Axis *xaxis,
				axis::Axis *yaxis = new CoordinateAxis(fYaxis, axis::fiphi), 
				axis::Axis *zaxis = new ValueAxis(fZaxis, fEntries), 
				int debug=0);

			//	booking
			virtual void book(DQMStore::IBooker&,
				std::string subsystem="Hcal", std::string aux="");

			//	filling
			virtual void fill(int, int);
			virtual void fill(int, double);
			virtual void fill(int, double, double);
			virtual void fill(int, int, int);
			virtual void fill(int, int, double);
			virtual void fill(double, int);
			virtual void fill(double, double);
			virtual void fill(double, double, double);

			//	any of the following 3 funcs must only be used with FEDs as
			//	X-axis and whatever coordinate as Yaxis
			//	there are no checks done on the axis!
			virtual void fill(int, HcalElectronicsId const&);
			virtual void fill(int, HcalElectronicsId const&, int);
			virtual void fill(int, HcalElectronicsId const&, double);

			virtual void fill(HcalDetId const&);
			virtual void fill(HcalDetId const&, double);
			virtual void fill(HcalDetId const&, double, double);

			virtual void fill(HcalElectronicsId const&);
			virtual void fill(HcalElectronicsId const&, double);
			virtual void fill(HcalElectronicsId const&, double, double);

			virtual void fill(HcalDetId const&, HcalElectronicsId const&);
			virtual void fill(HcalDetId const&, HcalElectronicsId const&, double);

			virtual void fill(HcalTrigTowerDetId const&);
			virtual void fill(HcalTrigTowerDetId const&, int);
			virtual void fill(HcalTrigTowerDetId const&, double);
			virtual void fill(HcalTrigTowerDetId const&, int, int);
			virtual void fill(HcalTrigTowerDetId const&, int, double);
			virtual void fill(HcalTrigTowerDetId const&, double, double);

			virtual void setBinContent(int, int, double);
			virtual void setBinContent(int, double, double);
			virtual void setBinContent(unsigned int, int, double);
			virtual void setBinContent(double, int, double);
			virtual void setBinContent(double, double, double);

			virtual void loadLabels(std::vector<std::string> const&);
			virtual void reset();

		protected:
			MonitorElement				*_me;
			Axis						*_xaxis;
			Axis						*_yaxis;
			Axis						*_zaxis;
	};
}

#endif
