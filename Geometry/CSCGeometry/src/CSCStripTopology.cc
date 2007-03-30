#include "Geometry/CSCGeometry/interface/CSCStripTopology.h"

CSCStripTopology::CSCStripTopology(int ns, float aw, float dh, float r, float aoff, float ymid ) :
   OffsetRadialStripTopology( ns, aw, dh, r, aoff, ymid ) {}

CSCStripTopology::~CSCStripTopology() {}


// op<< is not a member

#include <iostream>

std::ostream& operator<<( std::ostream& os, const CSCStripTopology& st )
{
  st.put( os ) << " isa " << static_cast<const OffsetRadialStripTopology&>( st );
  return os;
}
