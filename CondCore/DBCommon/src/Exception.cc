#include "CondCore/DBCommon/interface/Exception.h"
cond::Exception::Exception( const std::string& message ):cms::Exception("Conditions",message){}
cond::Exception::~Exception(){}
