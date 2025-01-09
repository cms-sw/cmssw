#ifndef CommonTools_Utils_TypedStringObjectMethodCaller_h
#define CommonTools_Utils_TypedStringObjectMethodCaller_h
/* \class TypedStringObjectMethodCaller
 * 
 * Object's method (or a chain of methods) caller functor with generic return-type, specified by string expression
 *
 */

#include "FWCore/Utilities/interface/EDMException.h"
#include "CommonTools/Utils/interface/parser/Exception.h"
#include "CommonTools/Utils/interface/parser/MethodChain.h"
#include "CommonTools/Utils/interface/parser/MethodChainGrammar.h"
#include "FWCore/Reflection/interface/ObjectWithDict.h"

template <typename T, typename R, bool DefaultLazyness = false>
struct TypedStringObjectMethodCaller {
  TypedStringObjectMethodCaller(const std::string expr, bool lazy = DefaultLazyness) : type_(typeid(T)) {
    using namespace boost::spirit::classic;
    reco::parser::MethodChainGrammar grammar(methodchain_, type_, lazy);
    const char* startingFrom = expr.c_str();
    try {
      if (!parse(startingFrom, grammar >> end_p, space_p).full) {
        throw edm::Exception(edm::errors::Configuration, "failed to parse \"" + expr + "\"");
      }
    } catch (reco::parser::BaseException& e) {
      throw edm::Exception(edm::errors::Configuration)
          << "MethodChainGrammer parse error:" << reco::parser::baseExceptionWhat(e) << " (char "
          << e.where - startingFrom << ")\n";
    }
  }

  R operator()(const T& t) const {
    edm::ObjectWithDict o(type_, const_cast<T*>(&t));
    edm::ObjectWithDict ret = methodchain_->value(o);
    return *static_cast<R*>(ret.address());
  }

private:
  reco::parser::MethodChainPtr methodchain_;
  edm::TypeWithDict type_;
};

#endif
