#include "PhysicsTools/Utilities/interface/MethodMap.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <Reflex/Base.h>
#include <iostream>
using namespace std;
using namespace ROOT::Reflex;
using namespace reco;

void MethodMap::fill( const Type & t, bool fillBase ) {
  using namespace method;
  if ( ! fillBase ) type_ = t;
  static std::map<string, method::retType> retTypeMap;
  if ( retTypeMap.size() == 0 ) {
    retTypeMap[ "double" ] = doubleType;
    retTypeMap[ "float" ] = floatType;
    retTypeMap[ "int" ] = intType;
    retTypeMap[ "unsigned int" ] = uIntType;
    retTypeMap[ "short" ] = shortType;
    retTypeMap[ "unsigned short" ] = uShortType;
    retTypeMap[ "long" ] = longType;
    retTypeMap[ "unsigned long" ] = uLongType;
    retTypeMap[ "char" ] = charType;
    retTypeMap[ "unsigned char" ] = uCharType;
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
    std::map<string, retType>::const_iterator f = retTypeMap.find( memberReturnType( mem ) );
    if ( f == retTypeMap.end() ) continue;
    map_[ methodName ] = make_pair( mem, f->second );
  }
  for( Base_Iterator b = t.Base_Begin(); b != t.Base_End(); ++ b ) {
    fill( b->ToType(), true );
  }
}

void MethodMap::print( std::ostream & out ) const{
  for( const_iterator i = begin(); i != end(); ++ i ) {
    out << memberReturnType( i->second.first ) << " " 
	<< type_.Name() << "::" 
	<< i->second.first.Name() << "()" << std::endl;
  }
}

string MethodMap::memberReturnType( const Member & mem ) {
  string name( mem.TypeOf().Name() );
  return name.substr( 0, name.find( '(' ) - 1 );
}
