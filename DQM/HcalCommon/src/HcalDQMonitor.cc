#include "DQM/HcalCommon/interface/HcalDQMonitor.h"

namespace hcaldqm
{
	//	Constructor
	HcalDQMonitor::HcalDQMonitor(edm::ParameterSet const& ps) : 
		_labels(ps.getUntrackedParameterSet("Labels"))
	{
		_mi.type		= ps.getUntrackedParameter<std::string>("mtype");
		_mi.runType		= ps.getUntrackedParameter<std::string>("runType");
		_mi.calibTypesAllowed	= ps.getUntrackedParameter<std::vector<int> >(
				"calibTypes");
		_mi.feds		= ps.getUntrackedParameter<std::vector<int> >(
				"FEDs");
		_mi.name		= ps.getUntrackedParameter<std::string>("name");
		_mi.debug		= ps.getUntrackedParameter<int>("debug");
		_mi.isGlobal	= ps.getUntrackedParameter<bool>("isGlobal");
		_mi.subsystem	= ps.getUntrackedParameter<std::string>("subsystem");

		_mi.currentCalibType	= -1;
		_mi.evsTotal			= 0;
		_mi.evsGood				= 0;
		_mi.evsPerLS			= 0;
		_mi.currentLS			= 0;
		_mi.isApplicable		= true;

		this->debug_("Calling Constructor");
	}

	//	Destructor
	/* virtual */ HcalDQMonitor::~HcalDQMonitor()
	{
		this->debug_("Calling Destructor");
	}
}
