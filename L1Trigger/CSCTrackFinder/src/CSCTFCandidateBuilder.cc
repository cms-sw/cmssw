#include <L1Trigger/CSCTrackFinder/src/CSCTFCandidateBuilder.h>

#include <DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h>

CSCTFCandidateBuilder::CSCTFCandidateBuilder(const edm::ParameterSet& pset)
{
  m_muonsorter = new CSCTFMuonSorter(pset);
}

void CSCTFCandidateBuilder::buildCandidates(const L1CSCTrackCollection* trks, 
					    std::vector<L1MuRegionalCand>* cands) const
{
  std::vector<L1MuRegionalCand> result;
  std::vector<csc::L1Track> stripped_tracks;
  
  L1CSCTrackCollection::const_iterator tmp_trk = trks->begin();

  std::cout << "INPUT TRACKS\n";
  
  L1CSCTrackCollection::const_iterator tt = trks->begin();

  for(; tt != trks->end(); tt++)
    {
      tt->first.Print();
      std::cout << "Track Stubs:\n";
      CSCCorrelatedLCTDigiCollection::DigiRangeIterator dd = tt->second.begin();

      for(; dd != tt->second.end(); dd++)
	{
	  CSCCorrelatedLCTDigiCollection::const_iterator lcts = (*dd).second.first;
	  for(; lcts != (*dd).second.second; lcts++)
	    {
	      std::cout << (*lcts) << std::endl;
	    }
	}
    }

  for(; tmp_trk != trks->end(); tmp_trk++)
    {
      stripped_tracks.push_back(tmp_trk->first);
    }

  result = m_muonsorter->run(stripped_tracks);

  cands->insert(cands->end(), result.begin(), result.end());
}
