#include "PhysicsTools/Utilities/interface/cutParser.h"
#include "PhysicsTools/Utilities/src/Grammar.h"
using namespace boost::spirit;

namespace reco {
  namespace parser {    
    bool cutParser( const std::string & value, const MethodMap& methods, SelectorPtr & sel ) {
      Grammar grammar( methods, sel );
      return parse( value.c_str(), grammar.use_parser<0>(), space_p ).full;
    }
  }
}
