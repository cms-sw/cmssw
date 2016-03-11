#ifndef Container_h
#define Container_h

/*
 *	file:		Container.h
 *	Author:		Viktor Khristenko
 *
 *	Description:
 *		Container Base class
 *		
 *
 */

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"
#include "DQM/HcalCommon/interface/Logger.h"

#include <vector>
#include <string>

namespace hcaldqm
{
	class Container
	{
		public:
			Container():
				_folder("HcalInfo"), _name("SOME_MONITOR")
			{}
			Container(std::string const& folder, std::string const &name):
				_folder(folder), _name(name)
			{}
			virtual ~Container() {}

			virtual void initialize(std::string const &folder, 
				std::string const &name, int debug=0)
			{
				_folder = folder;
				_name = name;
				_logger.set(_name, debug);
			}

		protected:
			std::string					_folder;
			std::string					_name;
			Logger						_logger;

	};
}


#endif








