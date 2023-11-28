#ifndef CommonTools_Utils_StringObjectFunction_h
#define CommonTools_Utils_StringObjectFunction_h
/* \class StringCutObjectSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: StringObjectFunction.h,v 1.4 2012/06/26 21:09:37 wmtan Exp $
 */
#include "FWCore/Utilities/interface/EDMException.h"
#include "CommonTools/Utils/interface/parser/ExpressionPtr.h"
#include "CommonTools/Utils/interface/parser/ExpressionBase.h"
#include "CommonTools/Utils/interface/expressionParser.h"
#include "FWCore/Reflection/interface/ObjectWithDict.h"

template <typename T, bool DefaultLazyness = false>
struct StringObjectFunction {
  StringObjectFunction(const std::string &expr, bool lazy = DefaultLazyness) : type_(typeid(T)) {
    if (!reco::parser::expressionParser<T>(expr, expr_, lazy)) {
      throw edm::Exception(edm::errors::Configuration, "failed to parse \"" + expr + "\"");
    }
  }
  StringObjectFunction(const reco::parser::ExpressionPtr &expr) : expr_(expr), type_(typeid(T)) {}
  double operator()(const T &t) const {
    edm::ObjectWithDict o(type_, const_cast<T *>(&t));
    return expr_->value(o);
  }

private:
  reco::parser::ExpressionPtr expr_;
  edm::TypeWithDict type_;
};

template <typename Object>
class sortByStringFunction {
public:
  sortByStringFunction(StringObjectFunction<Object> *f) : f_(f) {}
  ~sortByStringFunction() {}

  bool operator()(const Object *o1, const Object *o2) { return (*f_)(*o1) > (*f_)(*o2); }

private:
  StringObjectFunction<Object> *f_;
};

#endif
