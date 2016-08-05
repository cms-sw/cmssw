#include "L1Trigger/L1TCommon/interface/Tools.h"

namespace l1t{
void str2VecStr_(const std::string& aStr, const std::string& delim, std::vector<std::string>& aVec)
{
        if ( !parse ( aStr.c_str(),
        (
                  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( aVec ) ] % delim.c_str() )
        ), boost::spirit::classic::nothing_p ).full )
        {
                throw std::runtime_error ("Wrong value format: " + aStr);
        }

        for(auto it = aVec.begin(); it != aVec.end(); ++it)
        {
                while (*(it->begin()) == ' ' || *(it->begin()) == '\n')
                        it->erase(it->begin());
                while (*(it->end()-1) == ' ' || *(it->end()-1) == '\n')
            it->erase(it->end()-1);
        }
}

std::vector<std::string> str2VecStr_(const std::string& aStr, const std::string& delim)
{
        std::vector<std::string> aVec;

        if ( !parse ( aStr.c_str(),
        (
                  (  (*(boost::spirit::classic::anychar_p - delim.c_str() )) [boost::spirit::classic::push_back_a ( aVec ) ] % delim.c_str() )
        ), boost::spirit::classic::nothing_p ).full )
        {
                throw std::runtime_error ("Wrong value format: " + aStr);
        }

        for(auto it = aVec.begin(); it != aVec.end(); ++it)
        {
                while (*(it->begin()) == ' ' || *(it->begin()) == '\n')
                        it->erase(it->begin());
                while (*(it->end()-1) == ' ' || *(it->end()-1) == '\n')
            it->erase(it->end()-1);
        }

        return aVec;
}

unsigned int convertFromHexStringToInt(const std::string& aHexString)
{	
	std::map<std::string, int> hexnums;
	std::string strHexNums("0123456789ABCDEF");
        for(unsigned int i=0; i<strHexNums.size(); i++)
                hexnums[strHexNums.substr(i,1)] = i;
	unsigned int tempNum(0);
        for (unsigned int i=aHexString.size()-1; i>=2; i--)
        	tempNum += hexnums[aHexString.substr(i,1)]*pow(16,(aHexString.size()-1-i));
        
	return tempNum;
}



}//ns
