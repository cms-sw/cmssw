#ifndef PhysicsTools_Utilities_cutParset_h
#define PhysicsTools_Utilities_cutParset_h
#include "PhysicsTools/Utilities/src/SelectorPtr.h"
#include "PhysicsTools/Utilities/src/AnyObjSelector.h"
#include "PhysicsTools/Utilities/src/Grammar.h"
#include <string>

namespace reco {
  namespace parser {
    template<typename T>
    bool cutParser(const std::string & cut, SelectorPtr & sel) {
      bool justBlanks = true;
      for(std::string::const_iterator c = cut.begin(); c != cut.end(); ++c)
	if(*c != ' ') { justBlanks = false; break; }
      if(justBlanks) {
	sel = SelectorPtr(new AnyObjSelector);
	return true;
      } else {
	using namespace boost::spirit;
	Grammar grammar(sel, (const T *)(0));
	return parse(cut.c_str(), grammar.use_parser<0>() >> end_p, space_p).full;
      }
    } 
  }
}

#endif
