#ifndef PhysicsTools_Utilities_StringCutObjectSelector_h
#define PhysicsTools_Utilities_StringCutObjectSelector_h
/* \class StringCutObjectSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: StringCutObjectSelector.h,v 1.1 2007/10/31 14:08:00 llista Exp $
 */
#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/Utilities/src/SelectorPtr.h"
#include "PhysicsTools/Utilities/src/SelectorBase.h"
#include "PhysicsTools/Utilities/interface/cutParser.h"

template<typename T>
struct StringCutObjectSelector {
  StringCutObjectSelector(const std::string & cut) : 
    type_(ROOT::Reflex::Type::ByTypeInfo(typeid(T))) {
    if(! reco::parser::cutParser<T>(cut, select_)) {
      throw edm::Exception(edm::errors::Configuration,
			   "failed to parse \"" + cut + "\"");
    }
  }
  StringCutObjectSelector(const reco::parser::SelectorPtr & select) : 
    select_(select),
    type_(ROOT::Reflex::Type::ByTypeInfo(typeid(T))) {
  }
  bool operator()(const T & t) const {
    using namespace ROOT::Reflex;
    Object o(type_, const_cast<T *>(& t));
    return (*select_)(o);  
  }

private:
  reco::parser::SelectorPtr select_;
  ROOT::Reflex::Type type_;
};

#endif
