#ifndef ContainerS_h
#define ContainerS_h

#include "DQM/HcalCommon/interface/Container.h"
#include "DQM/HcalCommon/interface/Constants.h"

namespace hcaldqm
{
	using namespace constants;
	class ContainerS : public Container
	{
		public:
			ContainerS():
				Container()
			{}
			ContainerS(std::string const& folder, std::string const& name):
				Container(folder, name)
			{}
			virtual ~ContainerS() {}
			
			virtual void initialize(std::string const& folder, 
				std::string const& name, int debug=0)
			{
				_folder = folder;
				_qname = name;
				_logger.set(_qname, debug);
			}

			virtual void fill(std::string const& x)
			{
				_me->Fill((std::string&)x);
			}
			
			virtual void book(DQMStore::IBooker &ib,
				std::string subsystem="Hcal", std::string aux="")
			{
				ib.setCurrentFolder(subsystem+"/"+_folder+aux);
				_me = ib.bookString(_qname, "NameToStart");
			}

		protected:
			MonitorElement			*_me;
	};
}

#endif
















