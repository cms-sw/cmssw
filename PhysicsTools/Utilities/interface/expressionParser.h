#ifndef PhysicsTools_Utilities_expressionParset_h
#define PhysicsTools_Utilities_expressionParset_h
#include "PhysicsTools/Utilities/src/ExpressionPtr.h"
#include "PhysicsTools/Utilities/src/Grammar.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <string>

namespace reco {
  namespace parser {
    template<typename T>
    bool expressionParser( const std::string & value, ExpressionPtr & expr) {
      using namespace boost::spirit;
      Grammar grammar(expr, (const T*)(0));
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
