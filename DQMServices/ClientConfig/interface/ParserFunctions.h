#ifndef DQMServices_ClientConfig_ParserFunctions_h
#define DQMServices_ClientConfig_ParserFunctions_h

#include "xercesc/util/XercesDefs.hpp"
#include "xercesc/parsers/XercesDOMParser.hpp"

#include <string>

namespace qtxml {
  inline std::string _toString(const XMLCh* toTranscode) {
    std::string tmp(xercesc::XMLString::transcode(toTranscode));
    return tmp;
  }

  inline XMLCh* _toDOMS(std::string temp) {
    XMLCh* buff = xercesc::XMLString::transcode(temp.c_str());
    return buff;
  }

}  // namespace qtxml
#endif  // DQMServices_ClientConfig_ParserFunctions_h
