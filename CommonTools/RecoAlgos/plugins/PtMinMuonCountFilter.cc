/* \class PtMinMuonCountFilter
 *
 * Filters events if at least N muons above 
 * a pt cut are present
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "CommonTools/UtilAlgos/interface/PtMinSelector.h"

typedef ObjectCountFilter<
          reco::MuonCollection, 
          PtMinSelector
        >::type PtMinMuonCountFilter;
template <>
struct SelectorFillDescriptions<PtMinMuonCountFilter> {
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("muons"));
    desc.add<double>("ptMin", 0.);
    descriptions.add("ptMinMuonCountFilter", desc);
  }
};

DEFINE_FWK_MODULE( PtMinMuonCountFilter );
