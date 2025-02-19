#ifndef CommonTools_Utils_expressionParset_h
#define CommonTools_Utils_expressionParset_h
#include "CommonTools/Utils/src/ExpressionPtr.h"
#include "CommonTools/Utils/interface/Exception.h"
#include <Reflex/Type.h>
#include <string>

namespace reco {
  namespace parser {
  bool expressionParser(const Reflex::Type &t, const std::string & value, ExpressionPtr & expr, bool lazy) ;

  template<typename T>
  bool expressionParser( const std::string & value, ExpressionPtr & expr, bool lazy=false) {
    return reco::parser::expressionParser(Reflex::Type::ByTypeInfo(typeid(T)), value, expr, lazy);
  }

  }
}

#endif
