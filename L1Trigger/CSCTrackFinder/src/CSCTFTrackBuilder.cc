#include <L1Trigger/CSCTrackFinder/src/CSCTFTrackBuilder.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerContainer.h>


CSCTFTrackBuilder::CSCTFTrackBuilder(const edm::ParameterSet& pset)
{
  for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
    {
      for(int s = CSCTriggerNumbering::minTriggerSectorId(); 
	  s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s)
	{
	  my_SPs[e-1][s-1] = new CSCTFSectorProcessor(e, s, pset);
	}
    }
}

CSCTFTrackBuilder::~CSCTFTrackBuilder()
{
  for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
    {
      for(int s = CSCTriggerNumbering::minTriggerSectorId(); 
	  s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s)
	{
	  delete my_SPs[e-1][s-1];
	  my_SPs[e-1][s-1] = NULL;
	}
    }
}

void CSCTFTrackBuilder::buildTracks(const CSCCorrelatedLCTDigiCollection* lcts, std::vector<csc::L1Track>* trks)
{
  CSCCorrelatedLCTDigiCollection::DigiRangeIterator r_iter = lcts->begin();
  CSCTriggerContainer<CSCTrackStub> the_stubs;

  for(; r_iter != lcts->end(); r_iter++)
    {
      CSCCorrelatedLCTDigiCollection::Range r = lcts->get((*r_iter).first);
      CSCCorrelatedLCTDigiCollection::const_iterator ilct = r.first;
      for(; ilct != r.second; ilct++)
	{
	  the_stubs.push_back(CSCTrackStub((*ilct), (*r_iter).first));
	}
    }

  for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
    {
      for(int s = CSCTriggerNumbering::minTriggerSectorId(); 
	  s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s)
	{
	  CSCTriggerContainer<CSCTrackStub> current_e_s = the_stubs.get(e, s);
	  if(my_SPs[e-1][s-1]->run(current_e_s))
	    trks->insert(trks->end(), my_SPs[e-1][s-1]->tracks().get().begin(), 
			 my_SPs[e-1][s-1]->tracks().get().end());
	}
    }
}

