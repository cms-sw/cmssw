#include "CondCore/DBCommon/interface/ConnectStringParser.h"
#include "CondCore/DBCommon/interface/Exception.h"
static std::string sqlitedbsuffix(".db");
cond::ConnectStringParser::ConnectStringParser( const std::string& inputStr ):m_isLogical(false),m_inputStr(inputStr){
  if( (m_inputStr.find(':')==std::string::npos) && *(m_inputStr.begin())=='/' ){
    m_isLogical=true;
  }
  m_result.reserve(3);
  if(m_isLogical){
    this->parseLogical();
  }else{
    this->parsePhysical();
  }
}
cond::ConnectStringParser::~ConnectStringParser(){
  m_result.clear();
}
bool 
cond::ConnectStringParser::isLogical() const{
  return m_isLogical;
}
void
cond::ConnectStringParser::reset( const std::string& inputStr ){
  m_result.clear();
  m_inputStr=inputStr;
  m_isLogical=false;
  if( (m_inputStr.find(':')==std::string::npos) && *(m_inputStr.begin())=='/' ){
    m_isLogical=true;
  }
  if(m_isLogical){
    this->parseLogical();
  }else{
    this->parsePhysical();
  }
}
/**
   logical connect string format
   "/protocol/servicelevel/logicalschema"
*/
void
cond::ConnectStringParser::parseLogical(){
  size_t protocolStartPos=m_inputStr.find_first_not_of('/');
  size_t protocolEndPos=m_inputStr.find_first_of('/',protocolStartPos);
  std::string protocol=m_inputStr.substr(protocolStartPos,
					 protocolEndPos-protocolStartPos);
  if(protocol == "oracle" || protocol == "frontier" ){
    m_result.push_back(protocol+"://" );
  }else if(protocol == "sqlite"){
    m_result.push_back("sqlite_file://");
  }else{
    throw cond::Exception("unsupported protocol");
  }
  size_t serviceStartPos=m_inputStr.find_first_not_of('/',protocolEndPos);
  size_t serviceEndPos=m_inputStr.find_first_of('/',serviceStartPos);
  std::string serviceLevel=m_inputStr.substr(serviceStartPos,serviceEndPos-serviceStartPos);
  if(serviceLevel=="dev"){
    if(protocol=="oracle"){
      m_result.push_back("devdb10");
      //continue;?
    }
    if(protocol=="sqlite"){
      m_result.push_back("dev");
    }
    if(protocol=="frontier"){
      m_result.push_back("CoralDev");
    }
  }else if(serviceLevel=="int"){
    if(protocol=="oracle"){
      m_result.push_back("cms_orcoff_int2r");
      //continue;?
    }
    if(protocol=="sqlite"){
      m_result.push_back("int");
    }
    if(protocol=="frontier"){
      m_result.push_back("FrontierInt");
    }
  }else if(serviceLevel=="offlineprod"){
    if(protocol=="oracle"){
      m_result.push_back("cms_orcoff_int2r");
    }
    if(protocol=="sqlite"){
      m_result.push_back("offlineprod");
    }
    if(protocol=="frontier"){
      m_result.push_back("FrontierInt");
    }
  }else if(serviceLevel=="onlineprod"){
    if(protocol=="oracle"){
      m_result.push_back("orcon");
    }
    if(protocol=="sqlite"){
      m_result.push_back("onlineprod");
    }
    if(protocol=="frontier"){
      m_result.push_back("FrontierOn");
    }
  }else{
    throw cond::Exception("not supported service level");
  }
  size_t schemaStartPos=m_inputStr.find_first_not_of('/',serviceEndPos);
  std::string schema=m_inputStr.substr(schemaStartPos,m_inputStr.length()-schemaStartPos);
  if(protocol=="sqlite"){
    m_result.push_back(std::string("-")+schema+sqlitedbsuffix);
  }else{
    m_result.push_back(std::string("/")+schema );
  }
}
/**
   physical connect string format
   "protocol://service/schema"
   sqlite_file://path/service_schema.db
   or special case for sqlite
*/
void
cond::ConnectStringParser::parsePhysical(){
  size_t protocolStartPos=0;
  size_t protocolEndPos=m_inputStr.find_first_of(':',protocolStartPos);
  std::string protocol=m_inputStr.substr(protocolStartPos,protocolEndPos-protocolStartPos);
  size_t serviceStartPos,serviceEndPos;
  std::string service;
  if(protocol=="oracle" || "frontier" ){
    m_result.push_back(std::string("/")+protocol);
    serviceStartPos=m_inputStr.find_first_not_of('/',protocolEndPos+3);
    serviceEndPos=m_inputStr.find_first_of('/',serviceStartPos);
  }else if(protocol.find("sqlite") != std::string::npos){
    m_result.push_back("/sqlite");
    serviceStartPos=m_inputStr.find_last_not_of('/',protocolEndPos);
    serviceEndPos=m_inputStr.find_last_of('-',protocolEndPos);
  }else{
    throw cond::Exception("unsupported protocol");
  }
  service=m_inputStr.substr(serviceStartPos,serviceEndPos-serviceStartPos);
  if( (protocol=="oracle" && service=="devdb10") ||
      (protocol.find("sqlite") != std::string::npos && service=="dev") ||
      (protocol=="frontier" && service=="CoralDev")
      ){
    m_result.push_back("/dev");
  }else if( (protocol=="oracle" && service=="cms_orcoff_int2r") ||
	    (protocol.find("sqlite") != std::string::npos && service=="int") ||
	    (protocol=="frontier" && service=="FrontierInt")
	    ){
    m_result.push_back("/int");
  }else if( (protocol=="oracle" && service=="cms_orcoff") ||
	    (protocol.find("sqlite") != std::string::npos && service=="offlineprod") ||(protocol=="frontier" && service=="Frontier")
	    ){
    m_result.push_back("/offlineprod");
  }else if( (protocol=="oracle" && service=="orcon") ||
	    (protocol.find("sqlite") != std::string::npos && service=="onlineprod")||
	    (protocol=="frontier" && service=="FrontierOn")
	    ){
    m_result.push_back("/onlineprod");
  }
  size_t schemaStartPos;
  std::string schema;
  if(protocol.find("sqlite") != std::string::npos){
    schemaStartPos=m_inputStr.find_last_not_of('-',serviceEndPos);
    size_t schemaEndPos=m_inputStr.find(sqlitedbsuffix,serviceEndPos);
    schema=m_inputStr.substr(schemaStartPos,schemaEndPos-schemaStartPos);
  }else{
    schemaStartPos=m_inputStr.find_first_not_of('/',serviceEndPos);
    schema=m_inputStr.substr(schemaStartPos,m_inputStr.length()-schemaStartPos);
  }
  m_result.push_back(std::string("/")+schema);
}
std::string 
cond::ConnectStringParser::result() const{
  std::string result("");
  for(std::vector<std::string>::const_iterator it=m_result.begin(); it!=m_result.end(); ++it){
    result += *it;
  }
  return result;
}



