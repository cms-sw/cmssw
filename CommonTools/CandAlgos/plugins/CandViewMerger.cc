/* \class CandViewMerger
 * 
 * Producer of merged Candidate collection 
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef Merger<reco::CandidateView, reco::CandidateCollection> CandViewMerger;

template <>
void CandViewMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("src",
                                       {
                                           edm::InputTag("collection1"),
                                           edm::InputTag("collection2"),
                                       });
  descriptions.add("candViewMerger", desc);
}


DEFINE_FWK_MODULE(CandViewMerger);
