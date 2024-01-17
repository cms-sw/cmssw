/* \class CandRefMerger
 * 
 * Producer of merged Candidate reference collection 
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef Merger<reco::CandidateBaseRefVector> CandRefMerger;

template <>
void CandRefMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("src",
                                       {
                                           edm::InputTag("collection1"),
                                           edm::InputTag("collection2"),
                                       });
  descriptions.add("candRefMerger", desc);
}

DEFINE_FWK_MODULE(CandRefMerger);
