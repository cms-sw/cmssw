#include "L1Trigger/L1TCommon/interface/trigSystem.h"

namespace l1t{

trigSystem::trigSystem() : _isConfigured(false)
{

}

trigSystem::~trigSystem()
{
	;
}

void trigSystem::configureSystem(const std::string& l1HltKey, const std::string& subSysName)
{
        std::cout << "L1_HLT_key: " << l1HltKey << ", subsystem name: " << subSysName << std::endl;
        // TODO: get _sysId from JSON
	//_sysId = "ugmt";
        // TODO: get processors and roles from JSON
        // loop to add all roles and processors found in the JSON
	//addProcRole("processors", "ugmt_processor");

        // TODO: get subsystem key and subsystem RS key from DB using l1HltKey
        // TODO: get clobs from subsystem keys

        // this is for loading from clobs from DB
        // loop over clobs received
        // build DOM from clob
        //_xmlRdr.readContext(domElement, _sysId, *this);

        // this is for loading from xml config file
        _xmlRdr.readDOMFromFile("ugmt_top_config_p5.xml");
        _xmlRdr.buildGlobalDoc("TestKey1");
        _xmlRdr.readContexts("TestKey1", _sysId, *this);

        _isConfigured = true;
}


void trigSystem::addProcRole(const std::string& role, const std::string& processor)
{
	for(auto it=_procRole.begin(); it!=_procRole.end(); it++)
	{
		if ( it->second.compare(processor) == 0 && it->first.compare(role) != 0 )
			throw std::runtime_error ("Processor: " + processor + " already exists but with different role");
	}	

	_procRole[processor] = role;

	_roleProcs[role].push_back(processor);

	_procEnabled[processor] = true;

}

void trigSystem::addSetting(const std::string& type, const std::string& id, const std::string& value, const std::string& procRole)
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

std::map<std::string, setting> trigSystem::getSettings(const std::string& proccessor)
{
	if (!_isConfigured)
		throw std::runtime_error("trigSystem is not configured yet. First call the configureSystem method");
	std::map<std::string, setting> settings;
	std::vector<setting> vecSettings = _procSettings.at(proccessor);
	for(auto it=vecSettings.begin(); it!=vecSettings.end(); it++)
		settings.insert(std::pair<std::string, setting>(it->getId(), *it));

	return settings;
}

template <class varType> bool trigSystem::checkIdExistsAndSetSetting(std::vector<varType>& vec, const std::string& id, const std::string& value, const std::string& procRole)
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

std::map<std::string, mask> trigSystem::getMasks(const std::string& proccessor)
{
	if (!_isConfigured)
		throw std::runtime_error("trigSystem is not configured yet. First call the configureSystem method");

	std::map<std::string, mask> masks;
	std::vector<mask> vecMasks= _procMasks.at(proccessor);
	for(auto it=vecMasks.begin(); it!=vecMasks.end(); it++)
		masks.insert(std::pair<std::string, mask>(it->getId(), *it));

	return masks;
}

bool trigSystem::isMasked(const std::string& proccessor, const std::string& id)
{
	if (!_isConfigured)
		throw std::runtime_error("trigSystem is not configured yet. First call the configureSystem method");

	bool isMasked = false;
	std::vector<mask> vecMasks= _procMasks.at(proccessor);
	for(auto it=vecMasks.begin(); it!=vecMasks.end(); it++) 
	{
		if (it->getId() == id) 
		{
			isMasked = true;
			break;
		}
    }

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

bool trigSystem::isProcEnabled(const std::string& proccessor)
{
	if (!_isConfigured)
		throw std::runtime_error("trigSystem is not configured yet. First call the configureSystem method");

	return _procEnabled[proccessor];
}

}
