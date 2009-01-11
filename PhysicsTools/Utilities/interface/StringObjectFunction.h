#ifndef Utilities_StringObjectFunction_h
#define Utilities_StringObjectFunction_h
/* \class StringCutObjectSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: StringObjectFunction.h,v 1.2 2007/12/19 10:26:37 llista Exp $
 */
#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/Utilities/src/ExpressionPtr.h"
#include "PhysicsTools/Utilities/src/ExpressionBase.h"
#include "PhysicsTools/Utilities/interface/expressionParser.h"

template<typename T>
struct StringObjectFunction {
  StringObjectFunction(const std::string & expr) : 
    type_(Reflex::Type::ByTypeInfo(typeid(T))) {
    if(! reco::parser::expressionParser<T>(expr, expr_)) {
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
