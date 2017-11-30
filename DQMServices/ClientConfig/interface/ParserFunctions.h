#ifndef ParserFunctions_H
#define ParserFunctions_H

#include <string>
#include "xercesc/util/XMLString.hpp"

namespace qtxml{
	inline std::string _toString(const XMLCh *toTranscode){
		std::string tmp(xercesc::XMLString::transcode(toTranscode));
		return tmp;
	}

	inline XMLCh*  _toDOMS( std::string temp ){
		XMLCh* buff = xercesc::XMLString::transcode(temp.c_str());    
		return  buff;
	}

}

#endif
