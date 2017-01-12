#ifndef DQModule_h
#define DQModule_h

/*
 *	file:			DQModule.h
 *	Author:			Viktor Khristenko
 *	date:			13.10.2015
 */

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"
#include "DQM/HcalCommon/interface/Logger.h"

#include <string>
#include <vector>
#include <iostream>

namespace hcaldqm
{
	//	Module Types:
	//	1) Tasks - process every event
	//	2) Clients - get loaded into the Harvester and processed sequentially
	//		used only for Online/Offline World Harvesting. Prefer standalone
	//		Harvesters
	//	3) Harvester - Client Manager - per lumi processing
	enum ModuleType
	{
		fTask = 0,
		fHarvester = 1,
		fClient = 2,
		nModuleType = 3
	};

	enum ProcessingType
	{
		fOnline = 0,
		fOffline = 1,
		fLocal = 2,
		nProcessingType = 3
	};

	std::string const pTypeNames[nProcessingType] = {
		"Online", "Offline", "Local"
	};

	class DQModule
	{
		public:
			DQModule(edm::ParameterSet const&);
			virtual ~DQModule() {}

		protected:
			//	Member variables	
			//	@name - module's name
			//	@ptype - Processing Type
			//	@mtype - Module Type
			//	@ctype - Calibration Type of the Module. All we want is 0 or 1 
			std::string				_name;
			ModuleType				_mtype;
			ProcessingType			_ptype;
			int						_debug;

			int						_runkeyVal;
			std::string				_runkeyName;
			std::string				_subsystem;

			int						_evsTotal;
			int						_evsPerLS;
			int						_currentLS;
			int						_maxLS;
			Logger					_logger;
	};
}

#endif










