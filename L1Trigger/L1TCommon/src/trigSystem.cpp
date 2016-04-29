#include "L1Trigger/L1TCommon/interface/trigSystem.h"

namespace l1t{

trigSystem::trigSystem() : _isConfigured(false)
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
        _xmlRdr.buildGlobalDoc(key);
        _xmlRdr.readContexts(key, _sysId, *this);

        _isConfigured = true;
}


void trigSystem::addProcRole(const std::string& processor, const std::string& role)
{
	for(auto it=_procRole.begin(); it!=_procRole.end(); it++)
	{
		if ( it->second.compare(processor) == 0 && it->first.compare(role) != 0 )
			throw std::runtime_error ("Processor: " + processor + " already exists but with different role");
	}	
	
	//std::cout << "Adding processor: " << processor << std::endl;
	_procRole[processor] = role;

	_roleProcs[role].push_back(processor);

	_procEnabled[processor] = true;

}

void trigSystem::addProcCrate(const std::string& processor, const std::string& crate)
{
	_daqttcProcs[crate].push_back(processor);
}

void trigSystem::addSetting(const std::string& type, const std::string& id, const std::string& value, const std::string& procRole)
{
	//std::cout << "Adding setting: " << id << std::endl;
	bool applyOnRole, foundRoleProc(false);
	for(auto it=_procRole.begin(); it!=_procRole.end(); it++)
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
		if (!checkIdExistsAndSetSetting(_procSettings[procRole], id, value, procRole))
			_procSettings[procRole].push_back(setting(type, id, value, procRole));

	}
	else
	{
		for( auto it = _roleProcs[procRole].begin(); it != _roleProcs[procRole].end(); it++)
		{			
			if ( _procSettings.find(*it) != _procSettings.end() )
			{
				bool settingAlreadyExist(false);
				for(auto is = _procSettings.at(*it).begin(); is != _procSettings.at(*it).end(); is++)
				{
					if (is->getId().compare(id) == 0)
					{
						settingAlreadyExist = true;
						break;
					}					
				}
				if (!settingAlreadyExist)
					_procSettings.at(*it).push_back(setting(type, id, value, procRole));
			}
			else
				_procSettings[*it].push_back(setting(type, id, value, procRole));
		}

	}
}

void trigSystem::addSettingTable(const std::string& id, const std::string& columns, const std::string& types,  const std::vector<std::string>& rows, const std::string& procRole, const std::string& delim)
{
	bool applyOnRole, foundRoleProc(false);
	for(auto it=_procRole.begin(); it!=_procRole.end(); it++)
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
		if (!checkIdExistsAndSetSetting(_procSettings[procRole], id, columns, types, rows, procRole, delim))
			_procSettings[procRole].push_back(setting(id, columns, types, rows, procRole, delim));

	}
	else
	{
		for( auto it = _roleProcs[procRole].begin(); it != _roleProcs[procRole].end(); it++)
		{			
			if ( _procSettings.find(*it) != _procSettings.end() )
			{
				bool settingAlreadyExist(false);
				for(auto is = _procSettings.at(*it).begin(); is != _procSettings.at(*it).end(); is++)
				{
					if (is->getId().compare(id) == 0)
					{
						settingAlreadyExist = true;
						break;
					}					
				}
				if (!settingAlreadyExist)
					_procSettings.at(*it).push_back(setting(id, columns, types, rows, procRole, delim));
			}
			else
				_procSettings[*it].push_back(setting(id, columns, types, rows, procRole, delim));
		}

	}
}

std::map<std::string, setting> trigSystem::getSettings(const std::string& processor)
{
	if (!_isConfigured)
		throw std::runtime_error("trigSystem is not configured yet. First call the configureSystem method");
	if ( _procRole.find(processor) == _procRole.end() )
		throw std::runtime_error ("Processor " + processor + " was not found in the trigSystem object list");

	std::map<std::string, setting> settings;
	std::vector<setting> vecSettings = _procSettings.at(processor);
	for(auto it=vecSettings.begin(); it!=vecSettings.end(); it++)
		settings.insert(std::pair<std::string, setting>(it->getId(), *it));

	return settings;
}

bool trigSystem::checkIdExistsAndSetSetting(std::vector<setting>& vec, const std::string& id, const std::string& value, const std::string& procRole)
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

bool trigSystem::checkIdExistsAndSetSetting(std::vector<setting>& vec, const std::string& id, const std::string& columns, const std::string& types,  const std::vector<std::string>& rows, const std::string& procRole, const std::string& delim)
{
	bool found(false);
	for(auto it = vec.begin(); it != vec.end(); it++)
	{
		if (it->getId().compare(id) == 0)
		{
			found = true;
						
			std::string vecTypes, vecColumns;
		/*	if ( !parse ( std::string(columns+delim+" ").c_str(),
			(
				* ( ( boost::spirit::classic::anychar_p - delim.c_str() )[boost::spirit::classic::push_back_a ( vecColumns ) ] >> *boost::spirit::classic::space_p )
			), boost::spirit::classic::nothing_p ).full )
			{ 	
				throw std::runtime_error ("Wrong value format: " + columns);
			}

			if ( !parse ( std::string(types+delim+" ").c_str(),
			(
				* ( ( boost::spirit::classic::anychar_p - delim.c_str() )[boost::spirit::classic::push_back_a ( vecTypes ) ] >> *boost::spirit::classic::space_p )
			), boost::spirit::classic::nothing_p ).full )
			{ 	
				throw std::runtime_error ("Wrong value format: " + types);
			}
*/
			it->resetTableRows();
			it->setTableTypes(vecTypes);
			it->setTableColumns(vecColumns);
			for(auto ir=rows.begin(); ir!=rows.end(); ir++)
				it->addTableRow(*ir, delim);
		}
	}

	return found;
}

void trigSystem::addMask(const std::string& id, const std::string& procRole)
{
	bool applyOnRole, foundRoleProc(false);
	for(auto it=_procRole.begin(); it!=_procRole.end(); it++)
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
		_procMasks[procRole].push_back(mask(id, procRole));

	else
	{
		for( auto it = _roleProcs[procRole].begin(); it != _roleProcs[procRole].end(); it++)
		{			
			if ( _procMasks.find(*it) != _procMasks.end() )
			{
				bool maskAlreadyExist(false);
				for(auto is = _procMasks.at(*it).begin(); is != _procMasks.at(*it).end(); is++)
				{
					if (is->getId().compare(id) == 0)
					{
						maskAlreadyExist = true;
						break;
					}					
				}
				if (!maskAlreadyExist)
					_procMasks.at(*it).push_back(mask(id, procRole));
			}
			else
				_procMasks[*it].push_back(mask(id, procRole));
		}

	}
}

std::map<std::string, mask> trigSystem::getMasks(const std::string& processor)
{
	if (!_isConfigured)
		throw std::runtime_error("trigSystem is not configured yet. First call the configureSystem method");
	if ( _procRole.find(processor) == _procRole.end() )
		throw std::runtime_error ("Processor " + processor + " was not found in the trigSystem object list");
	
	std::map<std::string, mask> masks;
	std::vector<mask> vecMasks= _procMasks.at(processor);
	for(auto it=vecMasks.begin(); it!=vecMasks.end(); it++)
		masks.insert(std::pair<std::string, mask>(it->getId(), *it));

	return masks;
}

bool trigSystem::isMasked(const std::string& processor, const std::string& id)
{
	if (!_isConfigured)
		throw std::runtime_error("trigSystem is not configured yet. First call the configureSystem method");

	bool isMasked = false;
	std::vector<mask> vecMasks= _procMasks.at(processor);
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
	if ( _procRole.find(daqProc) == _procRole.end() && _daqttcProcs.find(daqProc) == _daqttcProcs.end())
		throw std::runtime_error("Cannot mask daq/processor " + daqProc + "! Not found in the system.");

	if ( _procRole.find(daqProc) != _procRole.end() )
		_procEnabled[daqProc] = false;
	else if ( _daqttcProcs.find(daqProc) != _daqttcProcs.end() )
	{
		for (auto it = _daqttcProcs[daqProc].begin(); it != _daqttcProcs[daqProc].end(); it++)
			_procEnabled[*it] = false;
	}
}

bool trigSystem::isProcEnabled(const std::string& processor)
{
	if (!_isConfigured)
		throw std::runtime_error("trigSystem is not configured yet. First call the configureSystem method");

	edm::LogInfo ("l1t::trigSystem::isProcEnabled") << "Returning " << _procEnabled[processor] << " for processor " << processor;
	return _procEnabled[processor];
}

}
