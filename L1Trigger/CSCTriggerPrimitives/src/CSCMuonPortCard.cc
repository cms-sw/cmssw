#include <L1Trigger/CSCTriggerPrimitives/src/CSCMuonPortCard.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

void CSCMuonPortCard::loadDigis(const CSCCorrelatedLCTDigiCollection& thedigis)
{
  clear();

  CSCCorrelatedLCTDigiCollection::DigiRangeIterator Citer;

  // First step is to put everything from the digi container into a trigger container.
  // This allows us to sort per BX more easily.
  for(Citer = thedigis.begin(); Citer != thedigis.end(); Citer++)
    {
      CSCCorrelatedLCTDigiCollection::const_iterator Diter = (*Citer).second.first;
      CSCCorrelatedLCTDigiCollection::const_iterator Dend = (*Citer).second.second;

      for(; Diter != Dend; Diter++)
	{
	  CSCTrackStub theStub((*Diter),(*Citer).first);
	  _stubs.push_back(theStub);
	}     
    }  
}

std::vector<CSCTrackStub> CSCMuonPortCard::sort(const unsigned& endcap, const unsigned& station, 
						const unsigned& sector, const unsigned& subsector, const int& bx)
{
  std::vector<CSCTrackStub> result;
  std::vector<CSCTrackStub>::iterator LCT;

  result = _stubs.get(endcap, station, sector, subsector, bx);

  /// Make sure no Quality 0 or non-valid LCTs come through the portcard.
  for(LCT = result.begin(); LCT != result.end(); LCT++)
    {
      if( !(LCT->getQuality() && LCT->isValid()) )
	result.erase(LCT, LCT);
    }

  if(result.size()) 
    {
      std::sort(result.begin(), result.end(), std::greater<CSCTrackStub>());
      if(result.size() > CSCConstants::maxStubs) /// Can only return maxStubs or less LCTs.
	result.erase(result.begin() + CSCConstants::maxStubs, result.end());


      /// go through the sorted list and label the LCTs with a sorting number
      unsigned i = 0;
      for(LCT = result.begin(); LCT != result.end(); LCT++)
	LCT->setMPCLink(++i);
    }  

  return result;
}
