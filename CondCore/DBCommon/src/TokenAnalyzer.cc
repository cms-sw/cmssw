#include "CondCore/DBCommon/interface/TokenAnalyzer.h"
#include "POOLCore/Token.h"
std::string 
cond::TokenAnalyzer::getFID(const std::string& strToken) const{
  pool::Token tok;
  return tok.fromString(strToken).dbID();
}
