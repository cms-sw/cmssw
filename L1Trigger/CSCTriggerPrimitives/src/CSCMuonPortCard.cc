#include "L1Trigger/CSCTriggerPrimitives/interface/CSCMuonPortCard.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"
#include <algorithm>

CSCMuonPortCard::CSCMuonPortCard() {}

CSCMuonPortCard::CSCMuonPortCard(const edm::ParameterSet& conf) {
  edm::ParameterSet mpcRun2Params = conf.getParameter<edm::ParameterSet>("mpcRun2");
  sort_stubs_ = mpcRun2Params.getParameter<bool>("sortStubs");
  drop_invalid_stubs_ = mpcRun2Params.getParameter<bool>("dropInvalidStubs");
  drop_low_quality_stubs_ = mpcRun2Params.getParameter<bool>("dropLowQualityStubs");
}

void CSCMuonPortCard::loadDigis(const CSCCorrelatedLCTDigiCollection& thedigis) {
  // Put everything from the digi container into a trigger container.
  // This allows us to sort per BX more easily.
  clear();

  CSCCorrelatedLCTDigiCollection::DigiRangeIterator Citer;

  for (Citer = thedigis.begin(); Citer != thedigis.end(); Citer++) {
    CSCCorrelatedLCTDigiCollection::const_iterator Diter = (*Citer).second.first;
    CSCCorrelatedLCTDigiCollection::const_iterator Dend = (*Citer).second.second;

    for (; Diter != Dend; Diter++) {
      csctf::TrackStub theStub((*Diter), (*Citer).first);
      stubs_.push_back(theStub);
    }
  }
}

std::vector<csctf::TrackStub> CSCMuonPortCard::sort(
    const unsigned endcap, const unsigned station, const unsigned sector, const unsigned subsector, const int bx) {
  std::vector<csctf::TrackStub> result;
  std::vector<csctf::TrackStub>::iterator LCT;

  result = stubs_.get(endcap, station, sector, subsector, bx);

  // Make sure no Quality 0 or non-valid LCTs come through the portcard.
  for (LCT = result.begin(); LCT != result.end(); LCT++) {
    if ((drop_invalid_stubs_ && !LCT->isValid()) || (drop_low_quality_stubs_ && LCT->getQuality() == 0))
      result.erase(LCT, LCT);
  }

  if (!result.empty()) {
    if (sort_stubs_)
      std::sort(result.begin(), result.end(), std::greater<csctf::TrackStub>());

    // Can return up to MAX_LCTS_PER_MPC (default 18) per bunch crossing.
    if (result.size() > CSCConstants::MAX_LCTS_PER_MPC)
      result.erase(result.begin() + CSCConstants::MAX_LCTS_PER_MPC, result.end());

    // Go through the sorted list and label the LCTs with a sorting number.
    unsigned i = 0;
    for (LCT = result.begin(); LCT != result.end(); LCT++)
      LCT->setMPCLink(++i);
  }

  return result;
}
