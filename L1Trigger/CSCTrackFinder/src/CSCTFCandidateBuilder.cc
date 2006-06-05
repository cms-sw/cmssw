#include <L1Trigger/CSCTrackFinder/src/CSCTFCandidateBuilder.h>

CSCTFCandidateBuilder::CSCTFCandidateBuilder(const edm::ParameterSet& pset)
{
  m_muonsorter = new CSCTFMuonSorter(pset);
}

void CSCTFCandidateBuilder::buildCandidates(const std::vector<csc::L1Track>* trks, 
					    std::vector<L1MuRegionalCand>* cands) const
{
  std::vector<L1MuRegionalCand> result;

  result = m_muonsorter->run(*trks);

  cands->insert(cands->end(), result.begin(), result.end());
}
