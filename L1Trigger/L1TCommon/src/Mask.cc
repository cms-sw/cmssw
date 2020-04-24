#include "L1Trigger/L1TCommon/interface/Mask.h"

namespace l1t{
	
Mask::Mask(std::string id, std::string procRole)
{
	id_ = id;
	port_ = boost::lexical_cast<int>(id.substr(id.find_last_not_of("0123456789")+1));
	procRole_ = procRole;
}

void Mask::setPort(std::string id)
{
	id_ = id;
	port_ = boost::lexical_cast<int>(id.substr(id.find_last_not_of("0123456789")+1));
}

}
