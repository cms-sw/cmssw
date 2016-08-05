#include "L1Trigger/L1TCommon/interface/TrigSystem.h"

namespace l1t{

TrigSystem::TrigSystem() : isConfigured_(false)
{
	logText_ = NULL;
}

TrigSystem::~TrigSystem()
{
	if (logText_)
	{
		std::cout << "Printing all logs\n" << *logText_;
		delete logText_;
	}
}

void TrigSystem::configureSystemFromFiles(const std::string& hwCfgFile, const std::string& topCfgFile, const std::string& key)
{
        // read hw description xml
        // this will set the sysId
        _xmlRdr.readDOMFromFile(hwCfgFile);
        _xmlRdr.readRootElement(*this);

        // read configuration xml files
        _xmlRdr.readDOMFromFile(topCfgFile);
        _xmlRdr.buildGlobalDoc(key, topCfgFile);
        _xmlRdr.readContexts(key, sysId_, *this);

        isConfigured_ = true;
}


void TrigSystem::addProcRole(const std::string& processor, const std::string& role)
{
	for(auto it=procRole_.begin(); it!=procRole_.end(); ++it)
	{
		if ( it->second.compare(processor) == 0 && it->first.compare(role) != 0 )
			throw std::runtime_error ("Processor: " + processor + " already exists but with different role");
	}	
	
	procRole_[processor] = role;

	roleProcs_[role].push_back(processor);

	procEnabled_[processor] = true;

}

void TrigSystem::addProcCrate(const std::string& processor, const std::string& crate)
{
	daqttcProcs_[crate].push_back(processor);
}

void TrigSystem::addSetting(const std::string& type, const std::string& id, const std::string& value, const std::string& procRole, const std::string& delim)
{
	bool applyOnRole, foundRoleProc(false);

	if (procRole_.find(procRole) != procRole_.end())
	{
		applyOnRole = false;
		foundRoleProc = true;
	}
	else if (roleProcs_.find(procRole) != roleProcs_.end())
	{
		applyOnRole = true;
		foundRoleProc = true;
	}

	if (!foundRoleProc)
		throw std::runtime_error ("Processor or Role " + procRole + " was not found in the map");

	if (!applyOnRole)
	{
		if (!checkIdExistsAndSetSetting_(procSettings_[procRole], id, value, procRole))
			procSettings_[procRole].push_back(Setting(type, id, value, procRole, logText_, delim));

	}
	else
	{
		for( auto it = roleProcs_[procRole].begin(); it != roleProcs_[procRole].end(); ++it)
		{			
			if ( procSettings_.find(*it) != procSettings_.end() )
			{
				bool SettingAlreadyExist(false);
				for(auto is = procSettings_.at(*it).begin(); is != procSettings_.at(*it).end(); ++is)
				{
					if (is->getId().compare(id) == 0)
					{
						SettingAlreadyExist = true;
						break;
					}					
				}
				if (!SettingAlreadyExist)
					procSettings_.at(*it).push_back(Setting(type, id, value, procRole, logText_, delim));
			}
			else
				procSettings_[*it].push_back(Setting(type, id, value, procRole, logText_, delim));
		}

	}
}

void TrigSystem::addSettingTable(const std::string& id, const std::string& columns, const std::string& types,  const std::vector<std::string>& rows, const std::string& procRole, const std::string& delim)
{
	bool applyOnRole, foundRoleProc(false);

	if (procRole_.find(procRole) != procRole_.end())
	{
		applyOnRole = false;
		foundRoleProc = true;
	}
	else if (roleProcs_.find(procRole) != roleProcs_.end())
	{
		applyOnRole = true;
		foundRoleProc = true;
	}
	if (!foundRoleProc)
		throw std::runtime_error ("Processor or Role " + procRole + " was not found in the map");

	if (!applyOnRole)
	{
		if (!checkIdExistsAndSetSetting_(procSettings_[procRole], id, columns, types, rows, procRole, delim))
			procSettings_[procRole].push_back(Setting(id, columns, types, rows, procRole, logText_, delim));

	}
	else
	{
		for( auto it = roleProcs_[procRole].begin(); it != roleProcs_[procRole].end(); ++it)
		{			
			if ( procSettings_.find(*it) != procSettings_.end() )
			{
				bool SettingAlreadyExist(false);
				for(auto is = procSettings_.at(*it).begin(); is != procSettings_.at(*it).end(); ++is)
				{
					if (is->getId().compare(id) == 0)
					{
						SettingAlreadyExist = true;
						break;
					}					
				}
				if (!SettingAlreadyExist)
					procSettings_.at(*it).push_back(Setting(id, columns, types, rows, procRole, logText_, delim));
			}
			else
				procSettings_[*it].push_back(Setting(id, columns, types, rows, procRole, logText_, delim));
		}

	}
}

std::map<std::string, Setting> TrigSystem::getSettings(const std::string& processor)
{
	if (!isConfigured_)
		throw std::runtime_error("TrigSystem is not configured yet. First call the configureSystem method");
	if ( procRole_.find(processor) == procRole_.end() )
		throw std::runtime_error ("Processor " + processor + " was not found in the TrigSystem object list");

	std::map<std::string, Setting> Settings;
	std::vector<Setting> vecSettings = procSettings_.at(processor);
	for(auto it=vecSettings.begin(); it!=vecSettings.end(); ++it)
		Settings.insert(std::pair<std::string, Setting>(it->getId(), *it));

	return Settings;
}

bool TrigSystem::checkIdExistsAndSetSetting_(std::vector<Setting>& vec, const std::string& id, const std::string& value, const std::string& procRole)
{
	bool found(false);
	for(auto it = vec.begin(); it != vec.end(); ++it)
	{
		if (it->getId().compare(id) == 0)
		{
			found = true;
			it->setValue(value);
			it->setProcRole(procRole);
		}
	}

	return found;
}

bool TrigSystem::checkIdExistsAndSetSetting_(std::vector<Setting>& vec, const std::string& id, const std::string& columns, const std::string& types,  const std::vector<std::string>& rows, const std::string& procRole, const std::string& delim)
{
	bool found(false);
	for(auto it = vec.begin(); it != vec.end(); ++it)
	{
		if (it->getId().compare(id) == 0)
		{
			found = true;
						
			it->resetTableRows();
			//it->setRowTypes(types);
			//it->setRowColumns(columns);
			for(auto ir=rows.begin(); ir!=rows.end(); ++ir)
				it->addTableRow(*ir, str2VecStr_(types, delim), str2VecStr_(columns, delim));

		}
	}

	return found;
}

void TrigSystem::addMask(const std::string& id, const std::string& procRole)
{
	bool applyOnRole, foundRoleProc(false);

	if (procRole_.find(procRole) != procRole_.end())
	{
		applyOnRole = false;
		foundRoleProc = true;
	}
	else if (roleProcs_.find(procRole) != roleProcs_.end())
	{
		applyOnRole = true;
		foundRoleProc = true;
	}

	if (!foundRoleProc)
		throw std::runtime_error ("Processor or Role " + procRole + " was not found in the map");

	if (!applyOnRole)
		procMasks_[procRole].push_back(Mask(id, procRole));

	else
	{
		for( auto it = roleProcs_[procRole].begin(); it != roleProcs_[procRole].end(); ++it)
		{			
			if ( procMasks_.find(*it) != procMasks_.end() )
			{
				bool MaskAlreadyExist(false);
				for(auto is = procMasks_.at(*it).begin(); is != procMasks_.at(*it).end(); ++is)
				{
					if (is->getId().compare(id) == 0)
					{
						MaskAlreadyExist = true;
						break;
					}					
				}
				if (!MaskAlreadyExist)
					procMasks_.at(*it).push_back(Mask(id, procRole));
			}
			else
				procMasks_[*it].push_back(Mask(id, procRole));
		}

	}
}

std::map<std::string, Mask> TrigSystem::getMasks(const std::string& processor)
{
	if (!isConfigured_)
		throw std::runtime_error("TrigSystem is not configured yet. First call the configureSystem method");
	if ( procRole_.find(processor) == procRole_.end() )
		throw std::runtime_error ("Processor " + processor + " was not found in the TrigSystem object list");
	
	std::map<std::string, Mask> Masks;
	std::vector<Mask> vecMasks= procMasks_.at(processor);
	for(auto it=vecMasks.begin(); it!=vecMasks.end(); ++it)
		Masks.insert(std::pair<std::string, Mask>(it->getId(), *it));

	return Masks;
}

bool TrigSystem::isMasked(const std::string& processor, const std::string& id)
{

	if (!isConfigured_)
		throw std::runtime_error("TrigSystem is not configured yet. First call the configureSystem method");

	bool isMasked = false;
	std::vector<Mask> vecMasks= procMasks_.at(processor);
	for(auto it=vecMasks.begin(); it!=vecMasks.end(); ++it) 
	{
		if (it->getId() == id) 
		{
			isMasked = true;
			break;
		}
    }

	edm::LogInfo ("l1t::TrigSystem::isMasked") << "Returning " << isMasked << " for processor " << processor << " and port " << id;
	return isMasked;
}

void TrigSystem::disableDaqProc(const std::string& daqProc)
{
	if ( procRole_.find(daqProc) == procRole_.end() && daqttcProcs_.find(daqProc) == daqttcProcs_.end())
		throw std::runtime_error("Cannot Mask daq/processor " + daqProc + "! Not found in the system.");

	if ( procRole_.find(daqProc) != procRole_.end() )
		procEnabled_[daqProc] = false;
	else if ( daqttcProcs_.find(daqProc) != daqttcProcs_.end() )
	{
		for (auto it = daqttcProcs_[daqProc].begin(); it != daqttcProcs_[daqProc].end(); ++it)
			procEnabled_[*it] = false;
	}
}

bool TrigSystem::isProcEnabled(const std::string& processor)
{
	if (!isConfigured_)
		throw std::runtime_error("TrigSystem is not configured yet. First call the configureSystem method");

	edm::LogInfo ("l1t::TrigSystem::isProcEnabled") << "Returning " << procEnabled_[processor] << " for processor " << processor;
	return procEnabled_[processor];
}

}
