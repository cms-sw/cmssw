#include "RecoJets/JetProducers/interface/PileupJetIdAlgo.h"
#include "CMGTools/External/interface/PileupJetIdentifierSubstructure.h"
#include "CMGTools/External/interface/PileupJetIdAlgoSubstructure.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/ValueMap.h"

namespace {
  namespace {
    PileupJetIdAlgo algo;
    PileupJetIdentifier ident;
    std::vector<StoredPileupJetIdentifierSubstructure> vec1;
    edm::ValueMap<StoredPileupJetIdentifierSubstructure> vmap1;
    edm::Wrapper<edm::ValueMap<StoredPileupJetIdentifierSubstructure> > vmapw1;
  }
}
