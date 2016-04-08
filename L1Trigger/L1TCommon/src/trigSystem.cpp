#include "L1Trigger/L1TCommon/interface/trigSystem.h"

namespace l1t{

trigSystem::trigSystem()
{

}

trigSystem::~trigSystem()
{
	;
}

void trigSystem::addProcRole(std::string role, std::string processor)
{
	for(auto it=_procRole.begin(); it!=_procRole.end(); it++)
	{
		if ( it->second.compare(processor) == 0 && it->first.compare(role) != 0 )
			throw std::runtime_error ("Processor: " + processor + " already exists but with different role");
	}	

	_procRole[processor] = role;

	_roleProcs[role].push_back(processor);
}

void trigSystem::addSetting(std::string type, std::string id, std::string value, std::string procRole)
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

std::map<std::string, setting> trigSystem::getSettings(std::string proccessor)
{
	std::map<std::string, setting> settings;
	std::vector<setting> vecSettings = _procSettings.at(proccessor);
	for(auto it=vecSettings.begin(); it!=vecSettings.end(); it++)
		settings.insert(std::pair<std::string, setting>(it->getId(), *it));

	return settings;
}

template <class varType> bool trigSystem::checkIdExistsAndSetSetting(std::vector<varType>& vec, std::string id, std::string value, std::string procRole)
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

void trigSystem::addMask(std::string id, std::string procRole)
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

std::map<std::string, mask> trigSystem::getMasks(std::string proccessor)
{
	std::map<std::string, mask> masks;
	std::vector<mask> vecMasks= _procMasks.at(proccessor);
	for(auto it=vecMasks.begin(); it!=vecMasks.end(); it++)
		masks.insert(std::pair<std::string, mask>(it->getId(), *it));

	return masks;
}

}
