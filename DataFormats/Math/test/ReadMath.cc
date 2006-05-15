#include "DataFormats/Math/test/ReadMath.h"
#include "DataFormats/Math/interface/Error.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>
using namespace std;
using namespace edm;

ReadMath::ReadMath( const ParameterSet& cfg ) :
  src( cfg.getParameter<string>( "src" ) ) {
}

void ReadMath::analyze( const Event & evt, const EventSetup & ) {
  typedef math::Error<6>::type Error;
  Handle<Error> err;
  evt.getByLabel( "src", err );
  for( int i = 0; i < 6; ++i ) {
    for( int j = 0; j < 6; ++j )
      cout << (*err)( i, j ) << " ";
    cout << endl;
  }
}
