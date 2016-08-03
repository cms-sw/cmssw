#ifndef __TrigSystem_h__
#define __TrigSystem_h__

#include <vector>
#include <string>
#include <map>

#include "./Setting.h"
#include "./Mask.h"
#include "L1Trigger/L1TCommon/interface/XmlConfigReader.h"

namespace l1t{

class TrigSystem
{
	public:
		TrigSystem();
		~TrigSystem();
		void configureSystemFromFiles(const std::string& hwCfgFile, const std::string& topCfgFile, const std::string& key);
		void setHwInfo(/*JSONConfigReader*/);
		void addProcRole(const std::string& processor, const std::string& role);
		void addProcCrate(const std::string& processor, const std::string& crate);
		void addSetting(const std::string& type, const std::string& id, const std::string& value, const std::string& procRole, const std::string& delim = "");
		void addSettingTable(const std::string& id, const std::string& columns, const std::string& types,  const std::vector<std::string>& rows, const std::string& procRole, const std::string& delim);
		void addMask(const std::string& id, const std::string& procRole);
		void disableDaqProc(const std::string& daqProc);
		const std::map<std::string, std::string>& getProcRole() { return procRole_; };
		const std::map<std::string, std::vector<std::string> >& getRoleProcs() { return roleProcs_; };
		std::map<std::string, Setting> getSettings (const std::string& processor);
		std::map<std::string, Mask> getMasks(const std::string& proccessor);
		bool isMasked(const std::string& proccessor, const std::string& id);
		bool isProcEnabled(const std::string& proccessor);
		std::string systemId() { return sysId_; };
		void setSystemId(const std::string id) { sysId_=id; };
		void setConfigured(const bool state=true) { isConfigured_=state; };
		//The setPrintAllLogs method should be called BEFORE calling the getters
		void setPrintAllLogs () { logText_= new  std::string(); };
	private:
		std::map<std::string, std::string> procRole_;
		std::map<std::string, std::vector<std::string> > roleProcs_;
		std::map<std::string, std::vector<std::string> > daqttcProcs_;
		std::map<std::string, std::vector<Setting> > procSettings_;
		std::map<std::string, std::vector<Mask> > procMasks_;
		std::map<std::string, bool> procEnabled_;

	    bool isConfigured_; 
		std::string sysId_;

		std::string* logText_;

		XmlConfigReader _xmlRdr;

		bool checkIdExistsAndSetSetting_(std::vector<Setting>& vec, const std::string& id, const std::string& value, const std::string& procRole);
		bool checkIdExistsAndSetSetting_(std::vector<Setting>& vec, const std::string& id, const std::string& columns, const std::string& types,  const std::vector<std::string>& rows, const std::string& procRole, const std::string& delim);
};

}

#endif

