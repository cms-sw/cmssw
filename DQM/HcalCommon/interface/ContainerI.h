#ifndef ContainerI_h
#define ContainerI_h

#include "DQM/HcalCommon/interface/Container.h"

namespace hcaldqm
{
	using namespace constants;
	class ContainerI : public Container
	{
		public:
			ContainerI():
				Container()
			{}
			ContainerI(std::string const& folder, std::string const& name):
				Container(folder, name)
			{}
			virtual ~ContainerI() {}
			
			virtual void initialize(std::string const& folder, 
				std::string const& name, int debug=0)
			{
				_folder = folder;
				_name = name;
				_logger.set(_name, debug);
			}

			virtual void fill(int x)
			{
				_me->Fill(x);
			}
			
			virtual void book(DQMStore::IBooker &ib,
				std::string subsystem="Hcal", std::string aux="")
			{
				ib.setCurrentFolder(subsystem+"/"+_folder +aux);
				_me = ib.bookInt(_name);
			}

		protected:
			MonitorElement			*_me;
	};
}

#endif
















