#include "DataFormats/ProtonReco/interface/ProtonTrack.h"
#include "DataFormats/ProtonReco/interface/ProtonTrackFwd.h"

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <vector>
#include <set>

namespace DataFormats_ProtonReco
{
  struct dictionary
  {
    reco::ProtonTrack pt;
    std::vector<reco::ProtonTrack> vec_pt;
    edm::Wrapper<std::vector<reco::ProtonTrack> > wrp_vec_pt;
    edm::RefProd<std::vector<reco::ProtonTrack> > rp_vec_pt;
    edm::Ref<std::vector<reco::ProtonTrack>,reco::ProtonTrack,edm::refhelper::FindUsingAdvance<std::vector<reco::ProtonTrack>,reco::ProtonTrack> > ref_vec_pt;
    edm::RefVector<std::vector<reco::ProtonTrack>,reco::ProtonTrack,edm::refhelper::FindUsingAdvance<std::vector<reco::ProtonTrack>,reco::ProtonTrack> > rv_vec_pt;
  };
}
