#ifndef __mask_h__
#define __mask_h__

#include <string>

//boost libraries
#include <boost/lexical_cast.hpp>

namespace l1t{
	
class mask
{
	public:
		mask(std::string id, std::string procRole);
		void setProcRole(std::string procRole) { _procRole = procRole; };
		void setPort(std::string id);
		std::string getProcRole() { return _procRole; };
		int getPort() { return _port; };
		std::string getId() {return _id;};

	private:
		int _port;
		std::string _procRole, _id;
};

}
#endif

