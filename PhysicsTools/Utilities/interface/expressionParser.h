#ifndef PhysicsTools_Utilities_expressionParset_h
#define PhysicsTools_Utilities_expressionParset_h
#include "PhysicsTools/Utilities/interface/MethodMap.h"
#include "PhysicsTools/Utilities/src/ExpressionPtr.h"
#include "PhysicsTools/Utilities/src/Grammar.h"
#include <string>

namespace reco {
  namespace parser {
    template<typename T>
    bool expressionParser( const std::string & value, ExpressionPtr & expr) {
      using namespace boost::spirit;
      Grammar grammar(reco::MethodMap::methods<T>(), expr);
      return parse(value.c_str(), grammar.use_parser<1>(), space_p).full;
    }
  }
}

#endif
