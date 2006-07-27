#include "PhysicsTools/UtilAlgos/interface/methods.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <iostream>
using namespace std;

namespace reco {
  namespace methods {
    void fill( methodMap & map, const ROOT::Reflex::Type & t ) {
      static std::map<string, retType> retTypeMap;
      if ( retTypeMap.size() == 0 ) {
	retTypeMap[ "double" ] = doubleType;
	retTypeMap[ "float" ] = floatType;
	retTypeMap[ "int" ] = intType;
	retTypeMap[ "unsigned int" ] = unsignedIntType;
	retTypeMap[ "char" ] = charType;
	retTypeMap[ "char int" ] = unsignedCharType;
	retTypeMap[ "bool" ] = boolType;
      }
      using namespace ROOT::Reflex;
      if ( ! t )  
	throw edm::Exception( edm::errors::ProductNotFound, "NoMatch" )
	  << "TypeID::className: No dictionary for class " << t.Name() << '\n';
      Member mem;
      for( size_t i = 0; i < t.FunctionMemberSize(); ++ i ) {
	mem = t.FunctionMemberAt( i );
	if ( mem.FunctionParameterSize( true ) != 0 ) continue;
	if ( mem.IsConstructor() ) continue;
	if ( mem.IsDestructor() ) continue;
	if ( mem.IsOperator() ) continue;
	if ( ! mem.IsPublic() ) continue;
	if ( mem.IsStatic() ) continue;
	if ( ! mem.TypeOf().IsConst() ) continue;
	string methodName = mem.Name();
	if ( methodName.substr( 0, 2 ) == "__" ) continue;
	string n = mem.TypeOf().Name();
	size_t p = n.find( '(' );
	string retName = n.substr( 0, p - 1 );
	std::map<string, retType>::const_iterator f = retTypeMap.find( retName );
	if ( f == retTypeMap.end() ) continue;
	map[ methodName ] = make_pair( mem, f->second );
      }
    }
  }
}
