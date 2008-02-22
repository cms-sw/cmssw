#ifndef PhysicsTools_Utilities_cutParset_h
#define PhysicsTools_Utilities_cutParset_h
#include "PhysicsTools/Utilities/src/SelectorPtr.h"
#include "PhysicsTools/Utilities/src/Grammar.h"
#include <string>

namespace reco {
  namespace parser {
    template<typename T>
    bool cutParser(const std::string & value, SelectorPtr & sel) {
      using namespace boost::spirit;
      Grammar grammar(sel, (const T *)(0));
      return parse(value.c_str(), grammar.use_parser<0>() >> end_p, space_p).full;
    } 
  }
}

#endif
