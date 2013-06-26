#include "DataFormats/TrackReco/interface/print.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <sstream>

using namespace std;

namespace reco {
  string print( const Track & trk, edm::Verbosity v ) {
    ostringstream out;
    if ( v > edm::Silent ) {
      out << "track parameters: " 
	  << " vtx = " << trk.vertex()
	  << " p = " << trk.momentum()
	  << endl;
    }
    if ( v >= edm::Detailed ) {
      out << "covariance" << endl;
      for ( int i = 0; i < 5; i++ ) {
	for ( int j = 0; j < 5; j++ ) {
	  out.precision(6); out.width(13); out << trk.covariance( i, j );
	}
	out << endl;
      }
      out << endl;
    }
    return out.str();
  }
}
