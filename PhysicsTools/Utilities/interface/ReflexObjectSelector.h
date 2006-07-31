#ifndef Utilities_ReflexObjectSelector_h
#define Utilities_ReflexObjectSelector_h
/* \class ReflexObjectSelector
 *
 *  Object selector template based on Reflex selector
 *
 * \author Luca Lista, INFN
 *
 */
#include "PhysicsTools/Utilities/interface/cutParser.h"
#include "PhysicsTools/Utilities/interface/MethodMap.h"

template<typename T>
struct ReflexObjectSelector {
  ReflexObjectSelector( const reco::parser::selector_ptr & select ) : 
    select_( select ) { }
  bool operator()( const T & t ) const { 
    return (*select_)( ROOT::Reflex::Object( reco::MethodMap::methods<T>().type(), const_cast<T *>( & t ) ) ); 
  }

private:
  reco::parser::selector_ptr select_;
};

#endif
