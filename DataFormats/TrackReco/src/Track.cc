#include "DataFormats/TrackReco/interface/Track.h"
using namespace reco;

Track::Track( float chi2, unsigned short ndof,  
	      int found, int lost, int invalid, 
	      const HelixParameters & helix  ) : 
  chi2_( chi2 ), ndof_( ndof ), 
  found_( found ), lost_( lost ), invalid_( invalid ),
  helix_( helix ) {
}


