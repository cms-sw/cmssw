#include "DataFormats/VertexReco/interface/Vertex.h"
// $Id: Vertex.cc,v 1.1 2005/11/22 14:00:31 llista Exp $
using namespace reco;
using namespace std;

Vertex::Vertex( const Point & p , const Error & err, double chi2, unsigned short ndof, size_t size ) :
  chi2_( chi2 ), ndof_( ndof ), position_( p ), error_( err ) {
  tracks_.reserve( size );
}
