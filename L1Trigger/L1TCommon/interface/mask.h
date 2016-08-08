#ifndef __mask_h__
#define __mask_h__

#include <string>

//boost libraries
#include <boost/lexical_cast.hpp>

namespace l1t{
	
class mask
{
	public:
		mask() {};
		mask(std::string id, std::string procRole);
		void setProcRole(std::string procRole) { procRole_ = procRole; };
		void setPort(std::string id);
		std::string getProcRole() { return procRole_; };
		unsigned getPort() { return port_; };
		std::string getId() {return id_;};

	private:
		unsigned port_;
		std::string procRole_, id_;
};

}
#endif

