#include "L1Trigger/CSCTrackFinder/src/CSCTFCandidateBuilder.h"

CSCTFCandidateBuilder::CSCTFCandidateBuilder(const edm::ParameterSet& pset) : m_muonsorter{pset} {}

void CSCTFCandidateBuilder::buildCandidates(const L1CSCTrackCollection* trks,
                                            std::vector<L1MuRegionalCand>* cands) const {
  std::vector<L1MuRegionalCand> result;
  CSCTriggerContainer<csc::L1Track> stripped_tracks;

  auto tmp_trk = trks->begin();

  for (; tmp_trk != trks->end(); tmp_trk++) {
    stripped_tracks.push_back(tmp_trk->first);
  }

  result = m_muonsorter.run(stripped_tracks);

  cands->insert(cands->end(), result.begin(), result.end());
}
