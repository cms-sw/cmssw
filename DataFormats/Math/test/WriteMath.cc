#include "DataFormats/Math/test/WriteMath.h"
#include "DataFormats/Math/interface/Error.h"
#include "FWCore/Framework/interface/Event.h"
using namespace edm;
using namespace std;
typedef math::Error<6>::type Error;

WriteMath::WriteMath( const ParameterSet& ) {
  produces<Error>();
}

void WriteMath::produce( Event & evt, const EventSetup & ) {
  double ee[] = { 1.00, 1.10, 1.20, 1.30, 1.40, 1.50,
		        2.00, 2.20, 2.30, 2.40, 2.50,
		              3.00, 3.30, 3.40, 3.50,
		                    4.00, 4.40, 4.50, 
		                          5.00, 5.50,
		                                6.00 };
  auto_ptr<Error> err( new Error );  
  int k = 0;
  for( int i = 0; i < 6; ++i )
    for( int j = 0; j <= i; ++j )
      (*err)( i, j ) = ee[ k++ ];
  
  evt.put( err );
}
