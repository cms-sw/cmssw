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
		void configureSystemFromFiles(const std::string& hwCfgFile, const std::string& topCfgFile, const std::string& key);
		void setHwInfo(/*JSONConfigReader*/);
		void addProcRole(const std::string& processor, const std::string& role);
		void addProcCrate(const std::string& processor, const std::string& crate);
		void addSetting(const std::string& type, const std::string& id, const std::string& value, const std::string& procRole);
		void addSettingTable(const std::string& id, const std::string& columns, const std::string& types,  const std::vector<std::string>& rows, const std::string& procRole, const std::string& delim);
		void addMask(const std::string& id, const std::string& procRole);
		void disableDaqProc(const std::string& daqProc);
		std::map<std::string, std::string> getProcRole() { return _procRole; };
		std::map<std::string, std::vector<std::string> > getRoleProcs() { return _roleProcs; };
		std::map<std::string, setting> getSettings (const std::string& processor);
		std::map<std::string, mask> getMasks(const std::string& proccessor);
		bool isMasked(const std::string& proccessor, const std::string& id);
		bool isProcEnabled(const std::string& proccessor);
		std::string systemId() { return _sysId; };
		void setSystemId(const std::string id) { _sysId=id; };
		void setConfigured(const bool state=true) { _isConfigured=state; };


	private:
		std::map<std::string, std::string> _procRole;
		std::map<std::string, std::vector<std::string> > _roleProcs;
		std::map<std::string, std::vector<std::string> > _daqttcProcs;
		std::map<std::string, std::vector<setting> > _procSettings;
		std::map<std::string, std::vector<mask> > _procMasks;
		std::map<std::string, bool> _procEnabled;

	        bool _isConfigured; 
		std::string _sysId;

		XmlConfigReader _xmlRdr;

		bool checkIdExistsAndSetSetting(std::vector<setting>& vec, const std::string& id, const std::string& value, const std::string& procRole);
		bool checkIdExistsAndSetSetting(std::vector<setting>& vec, const std::string& id, const std::string& columns, const std::string& types,  const std::vector<std::string>& rows, const std::string& procRole, const std::string& delim);
};

}

#endif

