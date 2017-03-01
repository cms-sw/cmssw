#ifndef ContainerD_h
#define ContainerD_h

#include "DQM/HcalCommon/interface/Container.h"

namespace hcaldqm
{
	using namespace constants;
	class ContainerD : public Container
	{
		public:
			ContainerD():
				Container()
			{}
			ContainerD(std::string const& folder, std::string const& name):
				Container(folder, name)
			{}
			virtual ~ContainerD() {}

			virtual void initialize(std::string const& folder,
				std::string const& name, int debug=0)
			{
				_folder = folder;
				_qname = name;
				_logger.set(_qname, debug);
			}

			virtual void fill(double x)
			{
				_me->Fill(x);
			}
			
			virtual void book(DQMStore::IBooker &ib,
				std::string subsystem="Hcal", std::string aux="")
			{
				ib.setCurrentFolder(subsystem+"/"+_folder+aux);
				_me = ib.bookFloat(_qname);
			}

		protected:
			MonitorElement			*_me;
	};
}

#endif
















