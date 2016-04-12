#ifndef __trigSystem_h__
#define __trigSystem_h__

#include <vector>
#include <string>
#include <map>

#include "./setting.h"
#include "./mask.h"
#include "L1Trigger/L1TCommon/interface/XmlConfigReader.h"

namespace l1t{

class trigSystem
{
	public:
		trigSystem();
		~trigSystem();
		void configureSystem(const std::string& l1HltKey, const std::string& subSysName);
		void setHwInfo(/*JSONConfigReader*/);
		void addProcRole(const std::string& role, const std::string& processor);
		void addSetting(const std::string& type, const std::string& id, const std::string& value, const std::string& procRole);
		void addMask(const std::string& id, const std::string& procRole);
		std::map<std::string, setting> getSettings (const std::string& processor);
		std::map<std::string, mask> getMasks(const std::string& proccessor);
		bool isMasked(const std::string& proccessor, const std::string& id);


	private:
		std::map<std::string, std::string> _procRole;
		std::map<std::string, std::vector<std::string> > _roleProcs;
		std::map<std::string, std::vector<setting> > _procSettings;
		std::map<std::string, std::vector<mask> > _procMasks;

                bool _isConfigured; // TODO: check this in getters
                std::string _sysId; // TODO: get from JSON

                XmlConfigReader _xmlRdr;

		template <class varType> bool checkIdExistsAndSetSetting(std::vector<varType>& vec, const std::string& id, const std::string& value, const std::string& procRole);
	
};

}

#endif

