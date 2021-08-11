#include "CommonTools/UtilAlgos/interface/AnySelector.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "CommonTools/UtilAlgos/interface/MinNumberSelector.h"
#include "CommonTools/UtilAlgos/interface/MaxNumberSelector.h"
#include "CommonTools/UtilAlgos/interface/AndSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"

namespace pat {

  typedef ObjectCountFilter<edm::View<reco::Candidate>,
                            AnySelector,
                            AndSelector<MinNumberSelector, MaxNumberSelector> >::type PATCandViewCountFilter;

}

using namespace pat;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATCandViewCountFilter);
