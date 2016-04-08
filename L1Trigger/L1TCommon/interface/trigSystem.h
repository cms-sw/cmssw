#ifndef __trigSystem_h__
#define __trigSystem_h__

#include <vector>
#include <string>
#include <map>

#include "./setting.h"
#include "./mask.h"

namespace l1t{
	
class trigSystem
{
	public:
		trigSystem();
		~trigSystem();
		void addProcRole(std::string role, std::string processor);
		void addSetting(std::string type, std::string id, std::string value, std::string procRole);
		void addMask(std::string id, std::string procRole);
		//std::vector<setting> getSettings(std::string proccessor);
		std::map<std::string, setting> getSettings (std::string processor); //write code add same for mask
		//std::vector<mask> getMasks(std::string proccessor);
		std::map<std::string, mask> getMasks(std::string proccessor);


	private:
		std::map<std::string, std::string> _procRole;
		std::map<std::string, std::vector<std::string> > _roleProcs;
		std::map<std::string, std::vector<setting> > _procSettings;
		std::map<std::string, std::vector<mask> > _procMasks;

		template <class varType> bool checkIdExistsAndSetSetting(std::vector<varType>& vec, std::string id, std::string value, std::string procRole);
	
};

}

#endif

