#include "CommonTools/Utils/interface/parser/ExpressionVarSetter.h"
#include "CommonTools/Utils/src/ExpressionVar.h"
#include "CommonTools/Utils/interface/returnType.h"
#include "CommonTools/Utils/interface/parser/Exception.h"
#include <string>
using namespace reco::parser;
using namespace std;

void ExpressionVarSetter::operator()(const char *begin, const char *end) const {
  //std::cerr << "ExpressionVarSetter: Pushed [" << std::string(begin,end) << "]" << std::endl;
  if (!methStack_.empty())
    push(begin, end);
  else if (!lazyMethStack_.empty())
    lazyPush(begin, end);
  else
    throw Exception(begin) << " Expression didn't parse neither hastily nor lazyly. This must not happen.\n";
}

void ExpressionVarSetter::push(const char *begin, const char *end) const {
  edm::TypeWithDict type = typeStack_.back();
  method::TypeCode retType = reco::typeCode(type);
  if (retType == method::invalid) {
    throw Exception(begin) << "member \"" << methStack_.back().methodName() << "\" has an invalid return type: \""
                           << methStack_.back().returnTypeName() << "\"";
  }
  if (!ExpressionVar::isValidReturnType(retType)) {
    throw Exception(begin) << "member \"" << methStack_.back().methodName() << "\" return type is \""
                           << methStack_.back().returnTypeName() << "\" which is not convertible to double.";
  }

  exprStack_.push_back(std::shared_ptr<ExpressionBase>(new ExpressionVar(methStack_, retType)));
  methStack_.clear();
  typeStack_.resize(1);
}

void ExpressionVarSetter::lazyPush(const char *begin, const char *end) const {
  exprStack_.push_back(std::shared_ptr<ExpressionBase>(new ExpressionLazyVar(lazyMethStack_)));
  lazyMethStack_.clear();
  typeStack_.resize(1);
}
