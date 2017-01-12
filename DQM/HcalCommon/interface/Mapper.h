#ifndef Mapper_h
#define Mapper_h

/*
 *	file:		Mapper.h
 *	Author:		Viktor Khristenko
 *
 *	Description:
 */

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"
#include "DQM/HcalCommon/interface/Constants.h"
#include "DQM/HcalCommon/interface/Logger.h"

#include <string>
#include <vector>
#include <sstream>

namespace hcaldqm
{
	namespace mapper
	{
		class Mapper
		{
			public:
				Mapper()
				{}
				virtual ~Mapper() {}

				virtual uint32_t getHash(HcalDetId const&) {return 0;}
				virtual uint32_t getHash(HcalElectronicsId const&) {return 0;}
				virtual uint32_t getHash(HcalTrigTowerDetId const&) {return 0;}

				virtual std::string getName(HcalDetId const&) {return "";}
				virtual std::string getName(HcalElectronicsId const&) 
				{return "";}
				virtual std::string getName(HcalTrigTowerDetId const&)
				{return "";}

			protected:
		};
	}
}

#endif





