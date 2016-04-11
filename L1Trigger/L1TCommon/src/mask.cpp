#include "L1Trigger/L1TCommon/interface/mask.h"

namespace l1t{
	
mask::mask(std::string id, std::string procRole)
{
	_id = id;
	_port = boost::lexical_cast<int>(id.substr(id.find_last_not_of("0123456789")+1));
	_procRole = procRole;
}

void mask::setPort(std::string id)
{
	_id = id;
	_port = boost::lexical_cast<int>(id.substr(id.find_last_not_of("0123456789")+1));
}

}
