#include "DataFormats/VertexReco/interface/Vertex.h"
// $Id: Vertex.cc,v 1.6 2005/11/21 12:55:16 llista Exp $
using namespace reco;
using namespace std;

Vertex::Vertex( double chi2, unsigned short ndof, 
		double x, double y, double z, const Error & err, 
		size_t size ) :
  chi2_( chi2 ), ndof_( ndof ), 
  position_(), error_( err ) {
  position_.get<0>() = x;
  position_.get<1>() = y;
  position_.get<2>() = z;
  tracks_.reserve( size );
}
