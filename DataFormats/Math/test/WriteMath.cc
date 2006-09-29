#include "DataFormats/Math/test/WriteMath.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "FWCore/Framework/interface/Event.h"
#include <vector>
using namespace edm;
using namespace std;
typedef math::XYZVector Vector;

WriteMath::WriteMath( const ParameterSet& ) {
  produces<vector<Vector> >();
}

void WriteMath::produce( Event & evt, const EventSetup & ) {
  auto_ptr<vector<Vector> > v( new vector<Vector> );  
  v->push_back( Vector( 1, 2, 3 ) );
  evt.put( v );
}
