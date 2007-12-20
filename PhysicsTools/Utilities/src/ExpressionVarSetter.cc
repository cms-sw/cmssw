#include "PhysicsTools/Utilities/src/ExpressionVarSetter.h"
#include "PhysicsTools/Utilities/src/ExpressionVar.h"
#include "PhysicsTools/Utilities/src/findMethod.h"
#include "PhysicsTools/Utilities/src/returnType.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <string>
#include <iostream>
using namespace reco::parser;
using namespace std;

void ExpressionVarSetter::operator()(const char * begin, const char* end) const {
  string methodName(begin, end);
  string::size_type endOfExpr = methodName.find_last_of(' ');
  if(endOfExpr != string::npos)
    methodName.erase(endOfExpr, methodName.size());
#ifdef BOOST_SPIRIT_DEBUG 
  BOOST_SPIRIT_DEBUG_OUT << "pushing variable: " << methodName << endl;
#endif
  ROOT::Reflex::Member mem = reco::findMethod(type_, methodName);
  method::TypeCode retType = reco::returnTypeCode(mem);
  if(retType == method::invalid)
    throw edm::Exception(edm::errors::Configuration)
      << "method \"" << mem.Name() 
      << "\" return type is not convertible to double\n";
  stack_.push_back(boost::shared_ptr<ExpressionBase>(new ExpressionVar(mem, retType)));
}
