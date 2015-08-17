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
      typedef reco::CutOnObject<T> CutType;
      
      edm::TypeID the_obj_type(typeid(T));
      edm::TypeID the_cut_type(typeid(CutType));

      std::stringstream func;
      
      bool justBlanks = true;
      for( auto chr : cut ) {
        if( !isspace(chr) ) {
          justBlanks = false;
          break;
        }
      }

      func << "bool eval(" << the_obj_type << " const& obj) const override final {\n";
      func << " return ( " << (justBlanks ? "true" : cut ) << " );\n";
      func << "}\n";

      const std::string strexpr = func.str();
      
      //std::cout << strexpr << std::endl;
      sel.reset( nullptr );
      try {
        reco::ExpressionEvaluator builder("CommonTools/CandUtils",the_cut_type.className().c_str(),strexpr.c_str());
        sel.reset( builder.expr<CutType>() );
      } catch( cms::Exception& e) {
        sel.reset( nullptr );
        throw edm::Exception(edm::errors::Configuration)
          <<"ExpressionEvaluator cut parser error: "<< e.what() << std::endl;
      }
      //std::cout << "made the builder" << std::endl;
      
      //std::cout << "made the expression" << std::endl;
      //std::cout << "set set = " << sel.get() << std::endl;
      
      return sel != nullptr;      
    }
    
    template<typename T>
    inline bool cutParser(const std::string & cut, SelectorPtr<T> & sel, bool lazy=false) {
      return reco::exprEval::cutParser(edm::TypeWithDict(typeid(T)), cut, sel, lazy);
    }
  }
}

#endif
