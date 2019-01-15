#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <vector>
#include <set>

namespace DataFormats_ProtonReco
{
  struct dictionary
  {
    reco::ForwardProton fp;
    std::vector<reco::ForwardProton> vec_fp;
    edm::Wrapper<std::vector<reco::ForwardProton> > wrp_vec_fp;
    edm::RefProd<std::vector<reco::ForwardProton> > rp_vec_fp;
    edm::Ref<std::vector<reco::ForwardProton>,reco::ForwardProton,edm::refhelper::FindUsingAdvance<std::vector<reco::ForwardProton>,reco::ForwardProton> > ref_vec_fp;
    edm::RefVector<std::vector<reco::ForwardProton>,reco::ForwardProton,edm::refhelper::FindUsingAdvance<std::vector<reco::ForwardProton>,reco::ForwardProton> > rv_vec_fp;
  };
}
