#ifndef __setting_h__
#define __setting_h__

#include <vector>
#include <string>

//boost libraries
#include <boost/lexical_cast.hpp>
#include <boost/spirit/include/classic.hpp>
#include <boost/spirit/include/classic_push_back_actor.hpp>

namespace l1t{
	
class setting
{
	public:
		setting(std::string type, std::string id, std::string value, std::string procRole);
		void setProcRole(std::string procRole) { _procRole = procRole; };
		void setValue(std::string value) {_value = value; };
		void setId(std::string id) { _id = id; } ;
		std::string getProcRole() { return _procRole; };
		std::string getValueAsStr() { return _value; };
		std::string getType() { return _type; };
		std::string getId() { return _id; } ;
		template <class varType> varType getValue();
		template <class varType> std::vector<varType> getVector(std::string delim = ",");
		~setting();

		setting& operator=(const setting& aSet);
	private:
		std::string _type, _id, _value, _procRole;
		

};

}
#endif

