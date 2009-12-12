#include "CommonTools/Utils/src/ExpressionVarSetter.h"
#include "CommonTools/Utils/src/ExpressionVar.h"
#include "CommonTools/Utils/src/returnType.h"
#include "CommonTools/Utils/interface/Exception.h"
#include <string>
using namespace reco::parser;
using namespace std;
using namespace Reflex;

void ExpressionVarSetter::operator()(const char * begin, const char* end) const {
  Type type = typeStack_.back();
  method::TypeCode retType = reco::typeCode(type);
  if(retType == method::invalid)
    throw  Exception(begin)
      << "member \"" << methStack_.back().method().Name() << "\" has an invalid return type: \"" 
      <<  methStack_.back().method().TypeOf().Name() << "\"";
  if(!ExpressionVar::isValidReturnType(retType))
     throw Exception(begin)
       << "member \"" << methStack_.back().method().Name() 
       << "\" return type is \"" << methStack_.back().method().TypeOf().Name() 
       << "\" which is not convertible to double.";
  
  exprStack_.push_back(boost::shared_ptr<ExpressionBase>(new ExpressionVar(methStack_, retType)));
  methStack_.clear();
  typeStack_.resize(1);
}
