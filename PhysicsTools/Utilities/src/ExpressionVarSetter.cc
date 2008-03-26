#include "PhysicsTools/Utilities/src/ExpressionVarSetter.h"
#include "PhysicsTools/Utilities/src/ExpressionVar.h"
#include "PhysicsTools/Utilities/interface/MethodMap.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <string>
#include <iostream>
using namespace reco::parser;
using namespace std;

void ExpressionVarSetter::operator()( const char * begin, const char* end ) const {
  string methodName( begin, end );
  string::size_type endOfExpr = methodName.find_last_of(' ');
  if( endOfExpr != string::npos )
    methodName.erase( endOfExpr, methodName.size() );
  reco::MethodMap::const_iterator m = methods_.find( methodName );
  if( m == methods_.end() )
    throw edm::Exception( edm::errors::Configuration, 
			  string( "unknown method \"" + methodName + "\"" ) );
#ifdef BOOST_SPIRIT_DEBUG 
  BOOST_SPIRIT_DEBUG_OUT << "pushing variable: " << methodName << endl;
#endif
  stack_.push_back( boost::shared_ptr<ExpressionBase>( new ExpressionVar( m->second ) ) );
}
