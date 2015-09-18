#ifndef CommonTools_Utils_cutParser_h
#define CommonTools_Utils_cutParser_h
#include "CommonTools/Utils/src/SelectorPtr.h"
#include "CommonTools/Utils/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include <string>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "CommonTools/Utils/interface/ExpressionEvaluator.h"
#include "CommonTools/Utils/interface/ExpressionEvaluatorTemplates.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cctype>

#include "CommonTools/Utils/interface/CutParserManager.h"

namespace reco {
  namespace parser {
    bool cutParser(const edm::TypeWithDict &t, const std::string & cut, SelectorPtr & sel, bool lazy) ;
    
    template<typename T>
    inline bool cutParser(const std::string & cut, SelectorPtr & sel, bool lazy=false) {
      return reco::parser::cutParser(edm::TypeWithDict(typeid(T)), cut, sel, lazy);
    }
  }
  namespace exprEval {
    template<typename T> 
    bool cutParser(const edm::TypeWithDict &t, const std::string & cut, SelectorPtr<T> & sel, bool lazy=false) {
      typedef reco::exprEval::ParsedCutManager<T,true>  LazyParser;
      typedef reco::exprEval::ParsedCutManager<T,false> Parser;
      if( lazy ) {
        edm::ObjectWithDict temp = t.construct();
        sel.reset(LazyParser::get()->getFunction(temp,cut));
      } else {
        sel.reset(Parser::get()->getFunction(t,cut));
      }
      return true;
    }
    
    template<typename T>
    inline bool cutParser(const std::string & cut, SelectorPtr<T> & sel, bool lazy=false) {
      return reco::exprEval::cutParser(edm::TypeWithDict(typeid(T)), cut, sel, lazy);
    }
  }
}

#endif
