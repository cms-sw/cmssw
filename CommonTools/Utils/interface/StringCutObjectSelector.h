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
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"

#include "CommonTools/Utils/interface/ExpressionEvaluator.h"
#include "CommonTools/Utils/interface/ExpressionEvaluatorTemplates.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <string>
#include <cctype>

template<typename T, bool DefaultLazyness=false>
struct StringCutObjectSelector {
  typedef reco::CutOnObject<T> CutType;
  
  StringCutObjectSelector(const std::string & cut, bool lazy=DefaultLazyness) :
  type_(typeid(T)){
    if(! reco::exprEval::cutParser<T>(cut, expr_select_, lazy)) {
      throw edm::Exception(edm::errors::Configuration,
			   "failed to parse \"" + cut + "\"");
    }
    
  }
  
  StringCutObjectSelector(const reco::exprEval::SelectorPtr<T> & select) : 
  expr_select_(select),
  type_(typeid(T)) {
  }

  bool operator()(const T & t) const {
    if( expr_select_ ) {
      return expr_select_->eval(t);
    }
    return false;
  }
  
  ~StringCutObjectSelector() { }

private:
  reco::exprEval::SelectorPtr<T> expr_select_; // this is not owned by us!!!!
  //reco::parser::SelectorPtr select_;
  edm::TypeWithDict type_;
};

#endif
