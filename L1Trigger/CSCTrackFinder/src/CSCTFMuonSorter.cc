#include <L1Trigger/CSCTrackFinder/interface/CSCTFMuonSorter.h>

CSCTFMuonSorter::CSCTFMuonSorter(const edm::ParameterSet& pset)
{
  m_minBX = pset.getUntrackedParameter<int>("MinBX",-3);
  m_maxBX = pset.getUntrackedParameter<int>("MaxBX",-3);
}

std::vector<L1MuRegionalCand> CSCTFMuonSorter::run(const CSCTriggerContainer<csc::L1Track>& tracks) const
{
  std::vector<L1MuRegionalCand> result;

  // First we sort and crop the incoming tracks based on their rank.
  for(int bx = m_minBX; bx <= m_maxBX; ++bx)
    {
      std::vector<csc::L1Track> tks = tracks.get(bx);
      std::sort(tks.begin(),tks.end(),std::greater<csc::L1Track>());
      tks.resize(4); // resize to max number of muons the MS can output
      
      std::vector<csc::L1Track>::iterator itr = tks.begin();
      std::vector<csc::L1Track>::const_iterator end = tks.end();
      for(; itr != end; itr++)
	{
	  unsigned gbl_phi = itr->localPhi() + ((itr->sector() - 1)*24) + 6; // for now, convert using this.. LUT in the future
	  itr->setPhiPacked(gbl_phi);
	  unsigned gbl_eta = itr->eta_packed() | (itr->endcap() == 1 ? 0 : 1) << (L1MuRegionalCand::ETA_LENGTH - 1);
	  itr->setEtaPacked(gbl_eta);
	  unsigned pt = 0, quality = 0;
	  decodeRank(itr->rank(), quality, pt);
	  itr->setQualityPacked(quality);
	  itr->setPtPacked(pt);
	  if(!itr->empty()) result.push_back(*itr);
	}
    }
  return result;
}


// This will change to use a look up table
void CSCTFMuonSorter::decodeRank(const unsigned& rank, unsigned& quality, 
				 unsigned& pt) const
{
  if(rank == 0)
    {
      quality = 0;
      pt = 0;
    }
  else
    {
      quality = rank >> L1MuRegionalCand::PT_LENGTH;
      pt = rank & ( (1<<L1MuRegionalCand::PT_LENGTH) - 1);
    }
}
