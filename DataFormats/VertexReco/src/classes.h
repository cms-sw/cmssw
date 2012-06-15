#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/NuclearInteraction.h"
#include <vector>
#include <utility>

namespace {
  struct dictionary {
    reco::Vertex rv1;
    std::vector<reco::Vertex> v1;
    edm::Wrapper<std::vector<reco::Vertex> > wc1;
    edm::Ref<std::vector<reco::Vertex> > r1;
    edm::RefProd<std::vector<reco::Vertex> > rp1;
    edm::RefVector<std::vector<reco::Vertex> > rvv1;

    reco::NuclearInteraction nrv1;
    std::vector<reco::NuclearInteraction> nv1;
    edm::Wrapper<std::vector<reco::NuclearInteraction> > nwc1;
    edm::Ref<std::vector<reco::NuclearInteraction> > nr1;
    edm::RefProd<std::vector<reco::NuclearInteraction> > nrp1;
    edm::RefVector<std::vector<reco::NuclearInteraction> > nrvv1;
  };
}
