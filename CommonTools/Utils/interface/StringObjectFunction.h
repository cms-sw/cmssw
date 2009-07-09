#ifndef CommonTools_Utils_StringObjectFunction_h
#define CommonTools_Utils_StringObjectFunction_h
/* \class StringCutObjectSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: StringObjectFunction.h,v 1.1 2009/02/24 14:10:19 llista Exp $
 */
#include "FWCore/Utilities/interface/EDMException.h"
#include "CommonTools/Utils/src/ExpressionPtr.h"
#include "CommonTools/Utils/src/ExpressionBase.h"
#include "CommonTools/Utils/interface/expressionParser.h"

template<typename T, bool DefaultLazyness=false>
struct StringObjectFunction {
  StringObjectFunction(const std::string & expr, bool lazy=DefaultLazyness) : 
    type_(Reflex::Type::ByTypeInfo(typeid(T))) {
    if(! reco::parser::expressionParser<T>(expr, expr_, lazy)) {
      throw edm::Exception(edm::errors::Configuration,
			   "failed to parse \"" + expr + "\"");
    }
  }
  StringObjectFunction(const reco::parser::ExpressionPtr & expr) : 
    expr_(expr),
    type_(Reflex::Type::ByTypeInfo(typeid(T))) {
  }
  double operator()(const T & t) const {
    using namespace Reflex;
    Object o(type_, const_cast<T *>(& t));
    return expr_->value(o);  
  }

private:
  reco::parser::ExpressionPtr expr_;
  Reflex::Type type_;
};

#endif
