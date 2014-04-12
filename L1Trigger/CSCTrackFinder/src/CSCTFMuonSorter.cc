#include <L1Trigger/CSCTrackFinder/interface/CSCTFMuonSorter.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

CSCTFMuonSorter::CSCTFMuonSorter(const edm::ParameterSet& pset)
{
  m_minBX = pset.getParameter<int>("MinBX");
  m_maxBX = pset.getParameter<int>("MaxBX");
}

std::vector<L1MuRegionalCand> CSCTFMuonSorter::run(const CSCTriggerContainer<csc::L1Track>& tracks) const
{
  std::vector<L1MuRegionalCand> result;

  // First we sort and crop the incoming tracks based on their rank.
  for(int bx = m_minBX - 6; bx <= m_maxBX - 6; ++bx) // switch back into signed BX
    {
      std::vector<csc::L1Track> tks = tracks.get(bx);
      std::sort(tks.begin(),tks.end(),std::greater<csc::L1Track>());
      if(tks.size() > 4) tks.resize(4); // resize to max number of muons the MS can output
      
      std::vector<csc::L1Track>::iterator itr = tks.begin();
      std::vector<csc::L1Track>::const_iterator end = tks.end();
      for(; itr != end; itr++)
	{
	  
	  
	  unsigned gbl_phi = itr->localPhi() + ((itr->sector() - 1)*24) + 6; // for now, convert using this.. LUT in the future
	  if(gbl_phi > 143) gbl_phi -= 143;	  
	  itr->setPhiPacked(gbl_phi & 0xff);
	  unsigned eta_sign = (itr->endcap() == 1 ? 0 : 1);

	  int gbl_eta = itr->eta_packed() | eta_sign << (L1MuRegionalCand::ETA_LENGTH - 1);

	  itr->setEtaPacked(gbl_eta & 0x3f);
	  unsigned pt = 0, quality = 0;
	  decodeRank(itr->rank(), quality, pt);

	  itr->setQualityPacked(quality & 0x3);
	  itr->setPtPacked(pt & 0x1f);

	  if(!itr->empty()) result.push_back(*itr);
	}
    }

  std::vector<L1MuRegionalCand>::const_iterator ittr = result.begin();
  unsigned ii = 1;
  for(; ittr != result.end(); ittr++)
    {
      LogDebug("CSCTFMuonSorter:run()") << "TRACK " << ii++ << ": Eta: " << ittr->etaValue() 
					<< " Phi: " << ittr->phiValue() << " Pt: " << ittr->ptValue()
					<< " Quality: " << ittr->quality() << " BX: " << ittr->bx();
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
