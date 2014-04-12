#include "RecoLuminosity/LumiProducer/interface/Exception.h"
lumi::Exception::Exception(const std::string& message,
			   const std::string& methodName,
			   const std::string& moduleName):m_message(message+" LUMI :\"" + methodName+"\" from \""+moduleName + "\" )") {}
lumi::nonCollisionException::nonCollisionException(
			   const std::string& methodName,
			   const std::string& moduleName):lumi::Exception("not a collision run" , methodName,moduleName) {}
lumi::invalidDataException::invalidDataException(
			   const std::string& message,			 
			   const std::string& methodName,
			   const std::string& moduleName):lumi::Exception("invalid data :"+message,methodName,moduleName) {}
lumi::noStableBeamException::noStableBeamException(
			   const std::string& message,			 
			   const std::string& methodName,
			   const std::string& moduleName):lumi::Exception("has no stable beam :"+message,methodName,moduleName) {}
lumi::duplicateRunInDataTagException::duplicateRunInDataTagException(
			   const std::string& message,			 
			   const std::string& methodName,
			   const std::string& moduleName):lumi::Exception("run already registered with the tag "+message,methodName,moduleName) {}
