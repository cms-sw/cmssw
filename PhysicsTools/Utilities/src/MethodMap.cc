#include "PhysicsTools/Utilities/src/MethodMap.h"
#include "PhysicsTools/Utilities/src/returnType.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <Reflex/Base.h>
#include <iostream>
using namespace std;
using namespace ROOT::Reflex;
using namespace reco;

void MethodMap::fill( const Type & t, bool fillBase ) {
  using namespace method;
  if ( ! fillBase ) type_ = t;
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
    TypeCode retType = reco::returnTypeCode(mem);
    map_[ methodName ] = make_pair(mem, retType);
  }
  for( Base_Iterator b = t.Base_Begin(); b != t.Base_End(); ++ b ) {
    fill( b->ToType(), true );
  }
  //cout << "TypeID::className: "  << t.Name() << endl;
  //print(cout);
}

