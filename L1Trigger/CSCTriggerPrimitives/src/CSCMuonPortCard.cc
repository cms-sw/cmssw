//-----------------------------------------------------------------------------
//
//   Class: CSCMuonPortCard
//
//   Description: 
//    Simulates the functionality of the Muon Port Card (MPC).  Each MPC
//    is responsible for 9 Trigger Mother Boards (TMBs).  It takes the up to
//    18 LCTs (2/TMB) in each (sub)sector every bunch crossing, sorts them,
//    selects up to three best, and puts them into an output collection.
//
//   Author List: Benn Tannenbaum 30 August 1999.
//                Based on code by Nick Wisniewski.
//
//   $Date: 2006/12/20 22:11:26 $
//   $Revision: 1.4 $
//
//   Modifications: Numerous later improvements by Jason Mumford and
//                  Slava Valuev (see cvs in ORCA).
//   Porting/reworking from ORCA by L. Gray (UF), June 2006.
//
//-----------------------------------------------------------------------------

#include <L1Trigger/CSCTriggerPrimitives/src/CSCMuonPortCard.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>

void CSCMuonPortCard::loadDigis(const CSCCorrelatedLCTDigiCollection& thedigis)
{
  // Put everything from the digi container into a trigger container.
  // This allows us to sort per BX more easily.
  clear();

  CSCCorrelatedLCTDigiCollection::DigiRangeIterator Citer;

  for (Citer = thedigis.begin(); Citer != thedigis.end(); Citer++) {
    CSCCorrelatedLCTDigiCollection::const_iterator Diter = (*Citer).second.first;
    CSCCorrelatedLCTDigiCollection::const_iterator Dend = (*Citer).second.second;

    for (; Diter != Dend; Diter++) {
      csctf::TrackStub theStub((*Diter), (*Citer).first);
      _stubs.push_back(theStub);
    }
  }
}

std::vector<csctf::TrackStub> CSCMuonPortCard::sort(const unsigned endcap, const unsigned station, 
						    const unsigned sector, const unsigned subsector, const int bx)
{
  std::vector<csctf::TrackStub> result;
  std::vector<csctf::TrackStub>::iterator LCT;

  result = _stubs.get(endcap, station, sector, subsector, bx);

  // Make sure no Quality 0 or non-valid LCTs come through the portcard.
  for (LCT = result.begin(); LCT != result.end(); LCT++) {
    if ( !(LCT->getQuality() && LCT->isValid()) )
      result.erase(LCT, LCT);
  }

  if (result.size()) {
    std::sort(result.begin(), result.end(), std::greater<csctf::TrackStub>());
    // Can only return maxStubs or less LCTs per bunch crossing.
    if (result.size() > CSCConstants::maxStubs)
      result.erase(result.begin() + CSCConstants::maxStubs, result.end());


    // Go through the sorted list and label the LCTs with a sorting number.
    unsigned i = 0;
    for (LCT = result.begin(); LCT != result.end(); LCT++)
      LCT->setMPCLink(++i);
  }

  return result;
}
