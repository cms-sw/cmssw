#include "RecoLuminosity/LumiProducer/interface/Exception.h"
lumi::Exception::Exception(const std::string& message,
			   const std::string& methodName,
			   const std::string& moduleName):m_message(message+" LUMI :\"" + methodName+"\" from \""+moduleName + "\" )") {}
lumi::nonCollisionException::nonCollisionException(
			   const std::string& methodName,
			   const std::string& moduleName):lumi::Exception("not a collision run" , methodName,moduleName) {}
