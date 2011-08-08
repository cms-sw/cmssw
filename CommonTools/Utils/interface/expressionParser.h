#ifndef CommonTools_Utils_expressionParset_h
#define CommonTools_Utils_expressionParset_h
#include "CommonTools/Utils/src/ExpressionPtr.h"
#include "CommonTools/Utils/src/Grammar.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <string>

namespace reco {
  namespace parser {
    template<typename T>
    bool expressionParser( const std::string & value, ExpressionPtr & expr, bool lazy=false) {
      using namespace boost::spirit::classic;
      Grammar grammar(expr, (const T*)(0), lazy);
      bool returnValue = false;
      const char* startingFrom = value.c_str();
      try {
         returnValue=parse(startingFrom, grammar.use_parser<1>() >> end_p, space_p).full;
      } catch(BaseException&e){
         throw edm::Exception(edm::errors::Configuration)<<"Expression parser error:"<<baseExceptionWhat(e)<<" (char "<<e.where-startingFrom<<")\n";
      }
      return returnValue;
    }
  }
}

#endif
