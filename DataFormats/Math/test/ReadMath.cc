#include "DataFormats/Math/test/ReadMath.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include <iostream>
using namespace std;
using namespace edm;

ReadMath::ReadMath( const ParameterSet& cfg ) :
  src( cfg.getParameter<InputTag>( "src" ) ) {
}

void ReadMath::analyze( const Event & evt, const EventSetup & ) {
  typedef math::XYZVector Vector;
  Handle<vector<Vector> > v;
  evt.getByLabel( src, v );
  cout << ">>> v = [ ";
  for( size_t i = 0; i < v->size(); ++ i )
    cout << (*v)[ i ] << ", ";
  cout << " ]" << endl;
}
