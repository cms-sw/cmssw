#ifndef CommonTools_Utils_StringObjectFunction_h
#define CommonTools_Utils_StringObjectFunction_h
/* \class StringCutObjectSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: StringObjectFunction.h,v 1.5 2012/08/03 18:08:09 wmtan Exp $
 */
#include "FWCore/Utilities/interface/EDMException.h"
#include "CommonTools/Utils/src/ExpressionPtr.h"
#include "CommonTools/Utils/src/ExpressionBase.h"
#include "CommonTools/Utils/interface/expressionParser.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"

template<typename T, bool DefaultLazyness=false>
struct StringObjectFunction {
  StringObjectFunction(const std::string & expr, bool lazy=DefaultLazyness) : 
    type_(typeid(T)) {
    if(! reco::parser::expressionParser<T>(expr, expr_, lazy)) {
      throw edm::Exception(edm::errors::Configuration,
			   "failed to parse \"" + expr + "\"");
    }
  }
  StringObjectFunction(const reco::parser::ExpressionPtr & expr) : 
    expr_(expr),
    type_(typeid(T)) {
  }
  double operator()(const T & t) const {
    edm::ObjectWithDict o(type_, const_cast<T *>(& t));
    return expr_->value(o);  
  }

private:
  reco::parser::ExpressionPtr expr_;
  edm::TypeWithDict type_;
};

#endif
