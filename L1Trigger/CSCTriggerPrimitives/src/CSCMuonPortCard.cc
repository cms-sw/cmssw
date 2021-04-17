#include "L1Trigger/CSCTriggerPrimitives/interface/CSCMuonPortCard.h"
#include "DataFormats/L1TMuon/interface/CSCConstants.h"
#include <algorithm>

CSCMuonPortCard::CSCMuonPortCard() {}

CSCMuonPortCard::CSCMuonPortCard(unsigned endcap, unsigned station, unsigned sector, const edm::ParameterSet& conf)
    : theEndcap(endcap), theStation(station), theSector(sector) {
  // Get min and max BX to sort LCTs in MPC.
  minBX_ = conf.getParameter<int>("MinBX");
  maxBX_ = conf.getParameter<int>("MaxBX");

  const auto& mpcParams = conf.getParameter<edm::ParameterSet>("mpcParam");
  sort_stubs_ = mpcParams.getParameter<bool>("sortStubs");
  drop_invalid_stubs_ = mpcParams.getParameter<bool>("dropInvalidStubs");
  drop_low_quality_stubs_ = mpcParams.getParameter<bool>("dropLowQualityStubs");
  max_stubs_ = mpcParams.getParameter<unsigned>("maxStubs");

  qualityControl_ = std::make_unique<LCTQualityControl>(endcap, station, sector, 1, 1, conf);

  const std::string eSign = endcap == 1 ? "+" : "-";
  vmeName_ = "VME" + eSign + std::to_string(theStation) + "/" + std::to_string(theSector);
}

void CSCMuonPortCard::clear() {
  stubs_.clear();
  selectedStubs_.clear();
}

void CSCMuonPortCard::loadLCTs(const CSCCorrelatedLCTDigiCollection& thedigis) {
  // clear the input and output collection
  clear();

  for (auto Citer = thedigis.begin(); Citer != thedigis.end(); Citer++) {
    const CSCDetId& detid((*Citer).first);
    const unsigned endcap = detid.endcap();
    const unsigned station = detid.station();
    const unsigned sector = detid.triggerSector();

    // select stubs by region
    if (endcap != theEndcap or station != theStation or sector != theSector)
      continue;

    // Put everything from the digi container into a trigger container.
    // This allows us to sort per BX more easily.
    for (auto Diter = (*Citer).second.first; Diter != (*Citer).second.second; Diter++) {
      stubs_.push_back(csctf::TrackStub((*Diter), (*Citer).first));
    }
  }
}

void CSCMuonPortCard::sortLCTs() {
  // sort the LCTs per BX and subsector
  for (int bx = minBX_; bx <= maxBX_; ++bx) {
    // station 1 case with all 10 degree chambers
    if (theStation == 1) {
      sortLCTs(1, bx);
      sortLCTs(2, bx);
    }
    // station 2,3,4 case with mixture of 10 and 20 degree chambers
    else {
      sortLCTs(0, bx);
    }
  }
}

void CSCMuonPortCard::sortLCTs(const unsigned subsector, const int bx) {
  // temporary vector
  std::vector<csctf::TrackStub> result = stubs_.get(theEndcap, theStation, theSector, subsector, bx);

  // pre-selection step
  for (auto LCT = result.begin(); LCT != result.end(); LCT++) {
    // step 1: no invalid stubs
    if (drop_invalid_stubs_ && !LCT->isValid()) {
      result.erase(LCT, LCT);
    }

    // step 2: no low-quality stubs
    if (drop_low_quality_stubs_ && LCT->getQuality() == 0) {
      result.erase(LCT, LCT);
    }
  }

  // sort+select
  if (!result.empty()) {
    // sort according to quality and CSCDetId
    if (sort_stubs_) {
      std::sort(result.begin(), result.end(), std::greater<csctf::TrackStub>());
    }

    // select up to MAX_LCTS_PER_MPC (default 18) per bunch crossing.
    const unsigned maxStubs = std::min(max_stubs_, unsigned(CSCConstants::MAX_LCTS_PER_MPC));
    if (result.size() > maxStubs) {
      result.erase(result.begin() + maxStubs, result.end());
    }

    // Go through the sorted list and label the LCTs with a sorting number.
    unsigned i = 0;
    for (auto LCT = result.begin(); LCT != result.end(); LCT++) {
      LCT->setMPCLink(++i);
    }

    // check if the MPC stubs are valid
    for (const auto& lct : result) {
      const CSCDetId& detid(lct.getDetId().rawId());
      const unsigned station(detid.station());
      const unsigned ring(detid.ring());
      qualityControl_->checkValid(*(lct.getDigi()), station, ring);
    }

    // now insert the temporary vector in the output collection
    selectedStubs_.insert(selectedStubs_.end(), result.begin(), result.end());
  }
}
