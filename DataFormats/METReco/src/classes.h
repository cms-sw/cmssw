#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace {
  namespace {
    edm::Wrapper<reco::MET> dummy2;
    edm::Wrapper<reco::METCollection> dummy3;
    edm::Wrapper< std::vector<reco::MET> > dummy4;
    std::vector<reco::MET> dummy5;
  }
}
