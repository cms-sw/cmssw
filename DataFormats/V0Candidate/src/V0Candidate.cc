#include "DataFormats/V0Candidate/interface/V0Candidate.h"

using namespace reco;

void V0Candidate::setRecoVertex( const Vertex & vtxIn ) {
  recoVertex = vtxIn;
  LeafCandidate::setVertex( vtxIn.position() );
}
