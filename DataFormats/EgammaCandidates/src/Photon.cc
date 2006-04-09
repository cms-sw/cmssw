// $Id: Photon.cc,v 1.2 2005/12/15 20:42:44 llista Exp $
#include "DataFormats/EgammaCandidates/interface/Photon.h"
using namespace reco;

Photon::Photon( const Vector & m,
		double z, short isolation, short pixelLines, bool hasSeed ) :
  momentum_( m ),  vtxZ_( z ), 
  isolation_( isolation ), pixelLines_( pixelLines ), hasSeed_( hasSeed ) {
}
