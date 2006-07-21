#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/PFReco/interface/PFCluster.h"
#include "DataFormats/PFReco/interface/PFRecHit.h"

namespace { 
  namespace {

    std::vector<reco::PFCluster> tsv3;
    std::vector<reco::PFRecHit>  tsv4;
    edm::Wrapper< std::vector<reco::PFRecHit> > PFHitProd;
  }
}
