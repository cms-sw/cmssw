#ifndef CommonTools_Utils_expressionParset_h
#define CommonTools_Utils_expressionParset_h
#include "CommonTools/Utils/src/ExpressionPtr.h"
#include "CommonTools/Utils/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include <string>

namespace reco {
  namespace parser {
  bool expressionParser(const edm::TypeWithDict &t, const std::string & value, ExpressionPtr & expr, bool lazy) ;

  template<typename T>
  bool expressionParser( const std::string & value, ExpressionPtr & expr, bool lazy=false) {
    return reco::parser::expressionParser(edm::TypeWithDict(typeid(T)), value, expr, lazy);
  }

  }
}

#endif
