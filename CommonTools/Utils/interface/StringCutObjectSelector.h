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

#include "CommonTools/Utils/interface/CutParserManager.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <string>
#include <cctype>

template<typename T, bool DefaultLazyness=false>
struct StringCutObjectSelector {
  typedef reco::CutOnObject<T> CutType;
  
  typedef reco::exprEval::ParsedCutManager<T,true>  LazyParser;
  typedef reco::exprEval::ParsedCutManager<T,false> Parser;

  StringCutObjectSelector(const std::string & cut, bool lazy=DefaultLazyness) :
  lazy_(lazy),
  cut_(cut),
  type_(typeid(T)),
  cut_manager_( lazy_ ? nullptr : Parser::get() ),
  lazy_cut_manager_( lazy_ ? LazyParser::get() : nullptr ),
  expr_select_( lazy_ ? nullptr : cut_manager_->getFunction(type_,cut_) )
   {
     /*
    if(! reco::exprEval::cutParser<T>(cut, expr_select_, lazy)) {
      throw edm::Exception(edm::errors::Configuration,
			   "failed to parse \"" + cut + "\"");
    }
    */
  }
  
  StringCutObjectSelector(reco::exprEval::SelectorPtr<T> select) : 
  lazy_(false),
  cut_(false),
  type_(typeid(T)),
  cut_manager_( nullptr ),
  lazy_cut_manager_( nullptr ),
  expr_select_(select)
  {
  }

  bool operator()(const T & t) const {
    if( lazy_ ) {
      edm::ObjectWithDict o(type_, const_cast<T *>(& t));
      return lazy_cut_manager_->getFunction(o,cut_)->eval(t);
    } else if( !lazy_ && expr_select_ ) {
      return expr_select_->eval(t);
    }
    return false;
  }
  
  ~StringCutObjectSelector() { }

private:  
  const bool lazy_;
  const std::string cut_;  
  edm::TypeWithDict type_;
  Parser const* const cut_manager_;
  LazyParser const* const lazy_cut_manager_;
  const reco::exprEval::SelectorPtr<T> expr_select_;  
};

#endif
