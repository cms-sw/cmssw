#ifndef CommonTools_Utils_StringCutObjectSelector_h
#define CommonTools_Utils_StringCutObjectSelector_h
/* \class StringCutObjectSelector
 *
 * \author Luca Lista, INFN ; L. Gray (FNAL) for exorcism of boost::spirit
 *
 * $Id: StringCutObjectSelector.h,v 1.4 2012/06/26 21:09:37 wmtan Exp $
 */
#include "FWCore/Utilities/interface/EDMException.h"
#include "CommonTools/Utils/src/SelectorPtr.h"
#include "CommonTools/Utils/src/SelectorBase.h"
#include "CommonTools/Utils/interface/cutParser.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"

#include "CommonTools/Utils/interface/ExpressionEvaluator.h"
#include "CommonTools/Utils/interface/ExpressionEvaluatorTemplates.h"

#include <iostream>
#include <sstream>
#include <string>
#include <cctype>

std::string demangle(const char* name);

template <class T>
std::string FriendlyType() {

    return demangle(typeid(T).name());
}

#ifdef __GNUG__
#include <cstdlib>
#include <memory>
#include <cxxabi.h>
std::string demangle(const char* name) {

    int status = -4; // some arbitrary value to eliminate the compiler warning

    // enable c++11 by passing the flag -std=c++11 to g++
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(name, NULL, NULL, &status),
        std::free
    };

    return std::string((status==0) ? res.get() : name);
}
#else
// does nothing if not g++
std::string demangle(const char* name) {
  return std::string(name);
}
#endif

template<typename T, bool DefaultLazyness=false>
struct StringCutObjectSelector {
  typedef reco::CutOnObject<T> CutType;
  
  StringCutObjectSelector(const std::string & cut, bool lazy=DefaultLazyness) :   
  type_(typeid(T)) {
    
    
    const std::string the_obj_type(FriendlyType<T>());
    const std::string the_cut_type(FriendlyType<CutType>());

    std::stringstream func;

    std::cout << " building a cut for: " << the_cut_type << " with cut: \"" << cut << "\"" << std::endl;

    const bool whiteSpacesOnly = std::all_of(cut.begin(),cut.end(),isspace);


    func << "bool test_bit(unsigned bits, unsigned ibit) const { return 0x1&(bits >> ibit); }\n";
    func << std::endl;
    func << "bool eval(" << the_obj_type << " const& cand) const override final {\n";
    func << " return ( " << (!whiteSpacesOnly ? cut : "true") << " );\n";
    func << "}\n";
    
    const std::string strexpr = func.str();

    std::cout << strexpr << std::endl;

    reco::ExpressionEvaluator builder("CommonTools/CandUtils",the_cut_type.c_str(),strexpr.c_str());
    std::cout << "made the builder" << std::endl;
    expr_select_ = builder.expr<CutType>();
    std::cout << "made the expression" << std::endl;
    std::cout << "set expr_select_ = " << expr_select_ << std::endl;
    /*
    if(! reco::parser::cutParser<T>(cut, select_, lazy)) {
      throw edm::Exception(edm::errors::Configuration,
			   "failed to parse \"" + cut + "\"");
    }
    */    
  }
  StringCutObjectSelector(const reco::parser::SelectorPtr & select) : 
  select_(select),
  type_(typeid(T)) {
  }
  bool operator()(const T & t) const {
    std::cout << "operator()" << std::endl;
    if( expr_select_ ) {
      std::cout << "calling eval" << std::endl;
      const bool temp = expr_select_->eval(t);
      std::cout << "called eval result=" << temp << std::endl;
      return temp;
    } else if ( select_ ) {
      edm::ObjectWithDict o(type_, const_cast<T *>(& t));
      return (*select_)(o);  
    }
    return false;
  }
  
private:
  const CutType* expr_select_; // this is not owned by us!!!!
  reco::parser::SelectorPtr select_;
  edm::TypeWithDict type_;
};

#endif
