
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/UserData.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include <vector>


namespace
{
  namespace
  {

    pat::UserHolder< bool > UHbool;

    pat::UserHolder< Vector3DBase< float, GlobalTag > > UHV3B;

    pat::UserHolder< edm::Ref< std::vector< reco::Vertex, std::allocator< reco::Vertex > >, reco::Vertex, edm::refhelper::FindUsingAdvance< std::vector< reco::Vertex, std::allocator< reco::Vertex > >, reco::Vertex > > > UHVTXRV;

    pat::UserHolder< edm::Ref< std::vector< pat::CompositeCandidate, std::allocator< pat::CompositeCandidate > >, pat::CompositeCandidate, edm::refhelper::FindUsingAdvance< std::vector< pat::CompositeCandidate, std::allocator< pat::CompositeCandidate > >, pat::CompositeCandidate > > > UHCCCRV;

  }
}
