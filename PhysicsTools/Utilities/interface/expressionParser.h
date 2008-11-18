#ifndef PhysicsTools_Utilities_expressionParset_h
#define PhysicsTools_Utilities_expressionParset_h
#include "PhysicsTools/Utilities/src/ExpressionPtr.h"
#include "PhysicsTools/Utilities/src/Grammar.h"
#include <string>

namespace reco {
  namespace parser {
    template<typename T>
    bool expressionParser( const std::string & value, ExpressionPtr & expr) {
      using namespace boost::spirit;
      Grammar grammar(expr, (const T*)(0));
      return parse(value.c_str(), grammar.use_parser<1>() >> end_p, space_p).full;
    }
  }
}

#endif
