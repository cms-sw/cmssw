// $Id: DSTPreOutput.cc,v 1.5 2005/07/14 11:45:30 llista Exp $

#include "PhysicsTools/CandAlgos/src/DSTPreOutput.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DSTTrack/interface/Track.h"
#include "DataFormats/DSTTrack/interface/BasicTrack.h"
#include "DataFormats/DSTTrack/interface/PixelTrack.h"
#include "DataFormats/DSTVertex/interface/Vertex.h"
#include "DataFormats/DSTMuon/interface/Muon.h"
#include <iostream> 
using namespace std;
using namespace edm;
using namespace dst;
using namespace phystools;

const string DSTPreOutput::names[] = { 
  "tracks", "pixeltracks", "basictracks", "vertices", "muons" 
};

DSTPreOutput::DSTPreOutput( const edm::ParameterSet & parms ) {
  int n = sizeof( names ) / sizeof( string );
  for( const string * tag = names; tag != names + n; ++ tag ) {
    try {
      tags[ * tag ] = parms.getParameter<vector<string> >( * tag ); 
    } catch ( edm::ParameterSetError e ) {
      cerr << ">>> no collection defined for " << tag  << endl;
    }
  }
}

template<typename T>
void DSTPreOutput::get( const Event & e, const vector<string> & v ) {
  for( vector<string>::const_iterator s = v.begin(); s != v.end(); ++ s )
    { Handle<vector<T> > h; e.getByLabel( * s, h ); }
}

void DSTPreOutput::analyze( const Event& e, const EventSetup& ) {
  get<Track     >( e, tags[ names[ tracks      ] ] );
  get<PixelTrack>( e, tags[ names[ pixeltracks ] ] );
  get<BasicTrack>( e, tags[ names[ basictracks ] ] );
  get<Vertex    >( e, tags[ names[ vertices    ] ] );
  get<Muon      >( e, tags[ names[ muons       ] ] );
}

// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
  
