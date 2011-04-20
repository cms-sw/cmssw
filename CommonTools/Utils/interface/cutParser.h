#ifndef CommonTools_Utils_cutParset_h
#define CommonTools_Utils_cutParset_h
#include "CommonTools/Utils/src/SelectorPtr.h"
#include "CommonTools/Utils/src/AnyObjSelector.h"
#include "CommonTools/Utils/src/Grammar.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <string>

namespace reco {
  namespace parser {
    template<typename T>
    bool cutParser(const std::string & cut, SelectorPtr & sel, bool lazy=false) {
      bool justBlanks = true;
      for(std::string::const_iterator c = cut.begin(); c != cut.end(); ++c) {
         if(*c != ' ') { justBlanks = false; break; }
      }
      if(justBlanks) {
         sel = SelectorPtr(new AnyObjSelector);
         return true;
      } else {
         using namespace boost::spirit::classic;
         Grammar grammar(sel, (const T *)(0), lazy);
         bool returnValue = false;
         const char* startingFrom =cut.c_str();
         try {
            returnValue = parse(startingFrom, grammar.use_parser<0>() >> end_p, space_p).full;
         } 
         catch(BaseException& e) {
            throw edm::Exception(edm::errors::Configuration)<<"Cut parser error:"<<baseExceptionWhat(e)<<" (char "<<e.where-startingFrom<<")\n";
         }
         return returnValue;
      }
    } 
  }
}

#endif
