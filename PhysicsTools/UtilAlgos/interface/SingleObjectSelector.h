#ifndef UtilAlgos_SingleObjectSelector_h
#define UtilAlgos_SingleObjectSelector_h
/* \class SingleObjectSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: SingleObjectSelector.h,v 1.1 2006/07/25 17:26:44 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/UtilAlgos/interface/ReflexSelector.h"
#include  <boost/shared_ptr.hpp>
#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/UtilAlgos/interface/cutParser.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

template<typename T>
struct SingleObjectSelector {
  SingleObjectSelector( const edm::ParameterSet & cfg ) : 
  type_( ROOT::Reflex::Type::ByTypeInfo( typeid( T ) ) ) {
  std::string cut = cfg.template getParameter<std::string>( "cut" );
  if( ! reco::parser::cutParser( cut, reco::methods::methods<T>(), select_ ) ) {
    throw edm::Exception( edm::errors::Configuration,
                          "failed to parse \"" + cut + "\"" );
  }
}
  bool operator()( const T & t ) const {
    using namespace ROOT::Reflex;
    Object o( type_, const_cast<T *>( & t ) );
    return (*select_)( o );  
  }

private:
  reco::parser::selector_ptr select_;
  ROOT::Reflex::Type type_;
};

#endif
