#ifndef L1Trigger_L1TCommon_Tools_h
#define L1Trigger_L1TCommon_Tools_h

#include <vector>
#include <map>

//boost libraries
#include <boost/lexical_cast.hpp>
#include <boost/spirit/include/classic.hpp>
#include <boost/spirit/include/classic_push_back_actor.hpp>

namespace l1t{

std::vector<std::string> str2VecStr_(const std::string& aStr, const std::string& delim);
void str2VecStr_(const std::string& aStr, const std::string& delim, std::vector<std::string>& aVec);
unsigned int convertFromHexStringToInt(const std::string& aHexString);
template <class varType> varType convertVariable(const std::string& aVar);

template <class varType> varType convertVariable(const std::string& aVar)
{
	varType temp;
	try{
		temp = boost::lexical_cast<varType>(aVar);
	}
	catch (std::exception& e)
	{
		std::map<std::string, int> hexnums;
        	std::string strHexNums("0123456789ABCDEFabcdef");
	       	for(unsigned int i=0; i<strHexNums.size(); i++)
	                hexnums[strHexNums.substr(i,1)] = i;
		if ( aVar.substr(0,2) == "0x" && aVar.substr(2,aVar.size()).find_first_not_of(strHexNums) == std::string::npos)
			temp = convertFromHexStringToInt(aVar);
		else
			throw std::runtime_error(std::string("Method convertVariable error: ") + e.what());
	}

	return temp;
}

}//ns

#endif
