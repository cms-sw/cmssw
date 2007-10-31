#include "PhysicsTools/Utilities/src/ExpressionVar.h"
#include <Reflex/Object.h>
using namespace reco::parser;
using namespace ROOT::Reflex;

double ExpressionVar::value( const Object & o ) const {
  using namespace method;
  Object ro = method_.first.Invoke( o );
  void * addr = ro.Address();
  double ret = 0;
  switch( method_.second ) {
  case( doubleType ) : ret = * static_cast<double         *>( addr ); break;
  case( floatType  ) : ret = * static_cast<float          *>( addr ); break;
  case( intType    ) : ret = * static_cast<int            *>( addr ); break;
  case( uIntType   ) : ret = * static_cast<unsigned int   *>( addr ); break;
  case( shortType  ) : ret = * static_cast<short          *>( addr ); break;
  case( uShortType ) : ret = * static_cast<unsigned short *>( addr ); break;
  case( longType   ) : ret = * static_cast<long           *>( addr ); break;
  case( uLongType  ) : ret = * static_cast<unsigned long  *>( addr ); break;
  case( charType   ) : ret = * static_cast<char           *>( addr ); break;
  case( uCharType  ) : ret = * static_cast<unsigned char  *>( addr ); break;
  case( boolType   ) : ret = * static_cast<bool           *>( addr ); break;
  };
  return ret;
}
