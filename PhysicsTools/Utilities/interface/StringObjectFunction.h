#ifndef Utilities_StringObjectFunction_h
#define Utilities_StringObjectFunction_h
/* \class StringCutObjectSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: StringObjectFunction.h,v 1.5 2007/10/23 07:33:35 llista Exp $
 */
#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/Utilities/src/ExpressionPtr.h"
#include "PhysicsTools/Utilities/src/ExpressionBase.h"
#include "PhysicsTools/Utilities/interface/expressionParser.h"

template<typename T>
struct StringObjectFunction {
  StringObjectFunction( const std::string & expr ) : 
    type_( ROOT::Reflex::Type::ByTypeInfo( typeid( T ) ) ) {
    if( ! reco::parser::expressionParser( expr, reco::MethodMap::methods<T>(), expr_ ) ) {
      throw edm::Exception( edm::errors::Configuration,
			    "failed to parse \"" + expr + "\"" );
    }
  }
  StringObjectFunction( const reco::parser::ExpressionPtr & expr ) : 
    expr_( expr ),
    type_( ROOT::Reflex::Type::ByTypeInfo( typeid( T ) ) ) {
  }
  double operator()( const T & t ) const {
    using namespace ROOT::Reflex;
    Object o( type_, const_cast<T *>( & t ) );
    return expr_->value( o );  
  }

private:
  reco::parser::ExpressionPtr expr_;
  ROOT::Reflex::Type type_;
};

#endif
