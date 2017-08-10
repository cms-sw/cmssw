#include "DataFormats/ProtonReco/interface/ProtonTrack.h"

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
    edm::View<reco::ProtonTrack> v_pt;
    edm::Ptr<reco::ProtonTrack> ptr_pt;
    std::vector< edm::Ptr<reco::ProtonTrack> > vec_ptr_pt;
    edm::Wrapper< std::vector<reco::ProtonTrack> > wrp_vec_pt;

    std::set<unsigned int> set_uint;
  };
}
