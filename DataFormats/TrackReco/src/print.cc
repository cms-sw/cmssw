#include "DataFormats/TrackReco/interface/print.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <sstream>

using namespace std;

namespace reco {
  string print( const Track & trk, verbosity v ) {
    ostringstream out;
    out << "track parameters" << endl;
    out << "(d_xy, phi_0, q/pT, d_z, p_z/p_T) = ";
    for (int i = 0; i < 5; i++) {
      out.precision(6); out.width(13); out<<trk.parameter(i);
    }
    out << endl;
    if ( v >= normal ) {
      out << "covariance" << endl;
      for (int i = 0; i < 5; i++) {
	for (int j = 0; j < 5; j++) {
	  out.precision(6); out.width(13); out<<trk.covariance(i, j);
	}
	out << endl;
      }
      out << endl;
    }
    if ( v >= detailed ) {
      out << "covariance (x,y,z,px,py,pz)" << endl;
      TrackBase::PosMomError err = trk.posMomError();
      for (int i = 0; i < 6; i++) {
	for (int j = 0; j < 6; j++) {
	  out.precision(6); out.width(13); out<< err(i,j);
	}
	out << endl;
      }
      out << endl;  
    }
    return out.str();
  }
}
