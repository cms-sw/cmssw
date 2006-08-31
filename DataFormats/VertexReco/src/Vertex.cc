#include "DataFormats/VertexReco/interface/Vertex.h"
// $Id: Vertex.cc,v 1.2 2005/12/15 20:42:52 llista Exp $
using namespace reco;
using namespace std;

Vertex::Vertex( const Point & p , const Error & err, double chi2, double ndof, size_t size ) :
  chi2_( chi2 ), ndof_( ndof ), position_( p ), error_( err ) {
  tracks_.reserve( size );
}
