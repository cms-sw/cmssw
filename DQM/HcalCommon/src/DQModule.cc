
#include "DQM/HcalCommon/interface/DQModule.h"

namespace hcaldqm
{
	DQModule::DQModule(edm::ParameterSet const& ps):
		_evsTotal(0), _evsPerLS(0)
	{
		_name = ps.getUntrackedParameter<std::string>("name", "Unknown_Module");
		_debug = ps.getUntrackedParameter<int>("debug", 0);
		_logger.set(_name, _debug);
		_runkeyVal = ps.getUntrackedParameter<int>("runkeyVal", 0);
		_runkeyName = ps.getUntrackedParameter<std::string>("runkeyName", 
			"pp_run");
		_subsystem = ps.getUntrackedParameter<std::string>("subsystem", "Hcal");

		bool mtype = ps.getUntrackedParameter<bool>("mtype", true);
		int ptype = ps.getUntrackedParameter<int>("ptype", 0);
		_maxLS = ps.getUntrackedParameter<int>("maxLS", 4000);
		if (mtype==true)
			_mtype = fTask;
		else
			_mtype = fClient;
		if (ptype==0)
			_ptype = fOnline;
		else if (ptype==1)
			_ptype = fOffline;
		else
			_ptype = fLocal;

		_logger.debug("Calling Constructor");
	}
}







