#ifndef UtilAlgos_SingleObjectSelector_h
#define UtilAlgos_SingleObjectSelector_h
/* \class SingleObjectSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: SingleObjectSelector.h,v 1.3 2006/08/04 11:56:42 llista Exp $
 */
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/Parser/interface/SelectorPtr.h"
#include "PhysicsTools/Parser/interface/SelectorBase.h"
#include "PhysicsTools/Parser/interface/cutParser.h"

template<typename T>
struct SingleObjectSelector {
  SingleObjectSelector( const edm::ParameterSet & cfg ) : 
    type_( ROOT::Reflex::Type::ByTypeInfo( typeid( T ) ) ) {
    std::string cut = cfg.template getParameter<std::string>( "cut" );
    if( ! reco::parser::cutParser( cut, reco::MethodMap::methods<T>(), select_ ) ) {
      throw edm::Exception( edm::errors::Configuration,
			    "failed to parse \"" + cut + "\"" );
    }
  }
  SingleObjectSelector( const std::string & cut ) : 
    type_( ROOT::Reflex::Type::ByTypeInfo( typeid( T ) ) ) {
    if( ! reco::parser::cutParser( cut, reco::MethodMap::methods<T>(), select_ ) ) {
      throw edm::Exception( edm::errors::Configuration,
			    "failed to parse \"" + cut + "\"" );
    }
  }
  SingleObjectSelector( const reco::parser::SelectorPtr & select ) : 
    select_( select ),
    type_( ROOT::Reflex::Type::ByTypeInfo( typeid( T ) ) ) {
  }
  bool operator()( const T & t ) const {
    using namespace ROOT::Reflex;
    Object o( type_, const_cast<T *>( & t ) );
    return (*select_)( o );  
  }

private:
  reco::parser::SelectorPtr select_;
  ROOT::Reflex::Type type_;
};

#endif
