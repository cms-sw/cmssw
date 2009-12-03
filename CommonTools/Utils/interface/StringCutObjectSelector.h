#ifndef CommonTools_Utils_StringCutObjectSelector_h
#define CommonTools_Utils_StringCutObjectSelector_h
/* \class StringCutObjectSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: StringCutObjectSelector.h,v 1.3 2009/01/11 23:37:33 hegner Exp $
 */
#include "FWCore/Utilities/interface/EDMException.h"
#include "CommonTools/Utils/src/SelectorPtr.h"
#include "CommonTools/Utils/src/SelectorBase.h"
#include "CommonTools/Utils/interface/cutParser.h"

template<typename T>
struct StringCutObjectSelector {
  StringCutObjectSelector(const std::string & cut) : 
    type_(Reflex::Type::ByTypeInfo(typeid(T))) {
    if(! reco::parser::cutParser<T>(cut, select_)) {
      throw edm::Exception(edm::errors::Configuration,
			   "failed to parse \"" + cut + "\"");
    }
  }
  StringCutObjectSelector(const reco::parser::SelectorPtr & select) : 
    select_(select),
    type_(Reflex::Type::ByTypeInfo(typeid(T))) {
  }
  bool operator()(const T & t) const {
    using namespace Reflex;
    Object o(type_, const_cast<T *>(& t));
    return (*select_)(o);  
  }

private:
  reco::parser::SelectorPtr select_;
  Reflex::Type type_;
};

#endif
