#include "PhysicsTools/Utilities/interface/expressionParser.h"
#include "PhysicsTools/Utilities/src/Grammar.h"
using namespace boost::spirit;

namespace reco {
  namespace parser {    
    bool expressionParser( const std::string & value, const MethodMap& methods, ExpressionPtr & expr ) {
      Grammar grammar( methods, expr );
      return parse( value.c_str(), grammar.use_parser<1>(), space_p ).full;
    }
  }
}
