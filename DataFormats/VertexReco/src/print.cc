#include "DataFormats/VertexReco/interface/print.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include <sstream>

using namespace std;

namespace reco {
  string print ( const Vertex & vtx, edm::Verbosity v ) {
    ostringstream out;
    if ( v > edm::Silent ) {
      out << "vertex position (x, y, z) = ( ";
      out.precision(6); out.width(13); out<< vtx.x();
      out.precision(6); out.width(13); out<< vtx.y();
      out.precision(6); out.width(13); out<< vtx.z();
      out << " )" << endl;
      //    if ( v >= normal ) {
      out << "error = " << endl;
      for (int i = 0; i < 2; i++) {
	for (int j = 0; j < 2; j++) {
	  out.precision(6); out.width(13); out<<vtx.covariance(i, j);
	}
	out << endl;
      }
      out << endl;
    }
    // if ( v >= edm::Detailed ) {
    //   print track weights
    //   print original and refitted track parameters
    // }
    return out.str();
  }
}
