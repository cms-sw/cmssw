#include "L1Trigger/L1TCommon/interface/trigSystem.h"

namespace l1t{

trigSystem::trigSystem() : isConfigured_(false)
{

}

trigSystem::~trigSystem()
{

}

void trigSystem::configureSystemFromFiles(const std::string& hwCfgFile, const std::string& topCfgFile, const std::string& key)
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


void trigSystem::addProcRole(const std::string& processor, const std::string& role)
{
	for(auto it=procRole_.begin(); it!=procRole_.end(); it++)
	{
		if ( it->second.compare(processor) == 0 && it->first.compare(role) != 0 )
			throw std::runtime_error ("Processor: " + processor + " already exists but with different role");
	}	
	
	//std::cout << "Adding processor: " << processor << std::endl;
	procRole_[processor] = role;

	roleProcs_[role].push_back(processor);

	procEnabled_[processor] = true;

}

void trigSystem::addProcCrate(const std::string& processor, const std::string& crate)
{
	daqttcProcs_[crate].push_back(processor);
}

void trigSystem::addSetting(const std::string& type, const std::string& id, const std::string& value, const std::string& procRole)
{
	//std::cout << "Adding setting: " << id << std::endl;
	bool applyOnRole, foundRoleProc(false);
	for(auto it=procRole_.begin(); it!=procRole_.end(); it++)
	{
		if (it->first.compare(procRole) == 0)
		{
			applyOnRole = false;
			foundRoleProc = true;
			break;
		}
		else if (it->second.compare(procRole) == 0)
		{
			applyOnRole = true;
			foundRoleProc = true;
			break;
		}
	}

	if (!foundRoleProc)
		throw std::runtime_error ("Processor or Role " + procRole + " was not found in the map");

	if (!applyOnRole)
	{
		if (!checkIdExistsAndSetSetting_(procSettings_[procRole], id, value, procRole))
			procSettings_[procRole].push_back(setting(type, id, value, procRole));

	}
	else
	{
		for( auto it = roleProcs_[procRole].begin(); it != roleProcs_[procRole].end(); it++)
		{			
			if ( procSettings_.find(*it) != procSettings_.end() )
			{
				bool settingAlreadyExist(false);
				for(auto is = procSettings_.at(*it).begin(); is != procSettings_.at(*it).end(); is++)
				{
					if (is->getId().compare(id) == 0)
					{
						settingAlreadyExist = true;
						break;
					}					
				}
				if (!settingAlreadyExist)
					procSettings_.at(*it).push_back(setting(type, id, value, procRole));
			}
			else
				procSettings_[*it].push_back(setting(type, id, value, procRole));
		}

	}
}

void trigSystem::addSettingTable(const std::string& id, const std::string& columns, const std::string& types,  const std::vector<std::string>& rows, const std::string& procRole, const std::string& delim)
{
	bool applyOnRole, foundRoleProc(false);
	for(auto it=procRole_.begin(); it!=procRole_.end(); it++)
	{
		if (it->first.compare(procRole) == 0)
		{
			applyOnRole = false;
			foundRoleProc = true;
			break;
		}
		else if (it->second.compare(procRole) == 0)
		{
			applyOnRole = true;
			foundRoleProc = true;
			break;
		}
	}

	if (!foundRoleProc)
		throw std::runtime_error ("Processor or Role " + procRole + " was not found in the map");

	if (!applyOnRole)
	{
		if (!checkIdExistsAndSetSetting_(procSettings_[procRole], id, columns, types, rows, procRole, delim))
			procSettings_[procRole].push_back(setting(id, columns, types, rows, procRole, delim));

	}
	else
	{
		for( auto it = roleProcs_[procRole].begin(); it != roleProcs_[procRole].end(); it++)
		{			
			if ( procSettings_.find(*it) != procSettings_.end() )
			{
				bool settingAlreadyExist(false);
				for(auto is = procSettings_.at(*it).begin(); is != procSettings_.at(*it).end(); is++)
				{
					if (is->getId().compare(id) == 0)
					{
						settingAlreadyExist = true;
						break;
					}					
				}
				if (!settingAlreadyExist)
					procSettings_.at(*it).push_back(setting(id, columns, types, rows, procRole, delim));
			}
			else
				procSettings_[*it].push_back(setting(id, columns, types, rows, procRole, delim));
		}

	}
}

std::map<std::string, setting> trigSystem::getSettings(const std::string& processor)
{
	if (!isConfigured_)
		throw std::runtime_error("trigSystem is not configured yet. First call the configureSystem method");
	if ( procRole_.find(processor) == procRole_.end() )
		throw std::runtime_error ("Processor " + processor + " was not found in the trigSystem object list");

	std::map<std::string, setting> settings;
	std::vector<setting> vecSettings = procSettings_.at(processor);
	for(auto it=vecSettings.begin(); it!=vecSettings.end(); it++)
		settings.insert(std::pair<std::string, setting>(it->getId(), *it));

	return settings;
}

bool trigSystem::checkIdExistsAndSetSetting_(std::vector<setting>& vec, const std::string& id, const std::string& value, const std::string& procRole)
{
	bool found(false);
	for(auto it = vec.begin(); it != vec.end(); it++)
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

bool trigSystem::checkIdExistsAndSetSetting_(std::vector<setting>& vec, const std::string& id, const std::string& columns, const std::string& types,  const std::vector<std::string>& rows, const std::string& procRole, const std::string& delim)
{
	bool found(false);
	for(auto it = vec.begin(); it != vec.end(); it++)
	{
		if (it->getId().compare(id) == 0)
		{
			found = true;
						
			it->resetTableRows();
			it->setTableTypes(types);
			it->setTableColumns(columns);
			for(auto ir=rows.begin(); ir!=rows.end(); ir++)
				it->addTableRow(*ir, delim);
		}
	}

	return found;
}

void trigSystem::addMask(const std::string& id, const std::string& procRole)
{
	bool applyOnRole, foundRoleProc(false);
	for(auto it=procRole_.begin(); it!=procRole_.end(); it++)
	{
		if (it->first.compare(procRole) == 0)
		{
			applyOnRole = false;
			foundRoleProc = true;
			break;
		}
		else if (it->second.compare(procRole) == 0)
		{
			applyOnRole = true;
			foundRoleProc = true;
			break;
		}
	}

	if (!foundRoleProc)
		throw std::runtime_error ("Processor or Role " + procRole + " was not found in the map");

	if (!applyOnRole)
		procMasks_[procRole].push_back(mask(id, procRole));

	else
	{
		for( auto it = roleProcs_[procRole].begin(); it != roleProcs_[procRole].end(); it++)
		{			
			if ( procMasks_.find(*it) != procMasks_.end() )
			{
				bool maskAlreadyExist(false);
				for(auto is = procMasks_.at(*it).begin(); is != procMasks_.at(*it).end(); is++)
				{
					if (is->getId().compare(id) == 0)
					{
						maskAlreadyExist = true;
						break;
					}					
				}
				if (!maskAlreadyExist)
					procMasks_.at(*it).push_back(mask(id, procRole));
			}
			else
				procMasks_[*it].push_back(mask(id, procRole));
		}

	}
}

std::map<std::string, mask> trigSystem::getMasks(const std::string& processor)
{
	if (!isConfigured_)
		throw std::runtime_error("trigSystem is not configured yet. First call the configureSystem method");
	if ( procRole_.find(processor) == procRole_.end() )
		throw std::runtime_error ("Processor " + processor + " was not found in the trigSystem object list");
	
	std::map<std::string, mask> masks;
	std::vector<mask> vecMasks= procMasks_.at(processor);
	for(auto it=vecMasks.begin(); it!=vecMasks.end(); it++)
		masks.insert(std::pair<std::string, mask>(it->getId(), *it));

	return masks;
}

bool trigSystem::isMasked(const std::string& processor, const std::string& id)
{
	if (!isConfigured_)
		throw std::runtime_error("trigSystem is not configured yet. First call the configureSystem method");

	bool isMasked = false;
	std::vector<mask> vecMasks= procMasks_.at(processor);
	for(auto it=vecMasks.begin(); it!=vecMasks.end(); it++) 
	{
		if (it->getId() == id) 
		{
			isMasked = true;
			break;
		}
    	}

	edm::LogInfo ("l1t::trigSystem::isMasked") << "Returning " << isMasked << " for processor " << processor << " and port " << id;
	return isMasked;
}

void trigSystem::disableDaqProc(const std::string& daqProc)
{
	if ( procRole_.find(daqProc) == procRole_.end() && daqttcProcs_.find(daqProc) == daqttcProcs_.end())
		throw std::runtime_error("Cannot mask daq/processor " + daqProc + "! Not found in the system.");

	if ( procRole_.find(daqProc) != procRole_.end() )
		procEnabled_[daqProc] = false;
	else if ( daqttcProcs_.find(daqProc) != daqttcProcs_.end() )
	{
		for (auto it = daqttcProcs_[daqProc].begin(); it != daqttcProcs_[daqProc].end(); it++)
			procEnabled_[*it] = false;
	}
}

bool trigSystem::isProcEnabled(const std::string& processor)
{
	if (!isConfigured_)
		throw std::runtime_error("trigSystem is not configured yet. First call the configureSystem method");

	edm::LogInfo ("l1t::trigSystem::isProcEnabled") << "Returning " << procEnabled_[processor] << " for processor " << processor;
	return procEnabled_[processor];
}

}
