#include "CondCore/DBCommon/interface/FipProtocolParser.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
cond::FipProtocolParser::FipProtocolParser(){}
cond::FipProtocolParser::~FipProtocolParser(){}

std::string 
cond::FipProtocolParser::getRealConnect(const std::string& fipConnect) const{
  std::string connect("sqlite_file:");
  std::string::size_type pos=fipConnect.find(':');
  std::string fipLocation=fipConnect.substr(pos+1);
  edm::FileInPath fip(fipLocation);
  connect.append( fip.fullPath() );
  return connect;
}
