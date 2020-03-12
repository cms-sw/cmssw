#include "L1Trigger/CSCTriggerPrimitives/interface/GEMCoPadProcessor.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <set>

//----------------
// Constructors --
//----------------

GEMCoPadProcessor::GEMCoPadProcessor(unsigned region, unsigned station, unsigned chamber, const edm::ParameterSet& copad)
    : theRegion(region), theStation(station), theChamber(chamber) {
  // Verbosity level, set to 0 (no print) by default.
  infoV = copad.getParameter<unsigned int>("verbosity");
  maxDeltaPad_ = copad.getParameter<unsigned int>("maxDeltaPad");
  maxDeltaRoll_ = copad.getParameter<unsigned int>("maxDeltaRoll");
  maxDeltaBX_ = copad.getParameter<unsigned int>("maxDeltaBX");
}

GEMCoPadProcessor::GEMCoPadProcessor() : theRegion(1), theStation(1), theChamber(1) {
  infoV = 0;
  maxDeltaPad_ = 0;
  maxDeltaRoll_ = 0;
  maxDeltaBX_ = 0;
}

void GEMCoPadProcessor::clear() { gemCoPadV.clear(); }

std::vector<GEMCoPadDigi> GEMCoPadProcessor::run(const GEMPadDigiCollection* in_pads) {
  // Build coincidences
  for (auto det_range = in_pads->begin(); det_range != in_pads->end(); ++det_range) {
    const GEMDetId& id = (*det_range).first;

    // same chamber (no restriction on the roll number)
    if (id.region() != theRegion or id.station() != theStation or id.chamber() != theChamber)
      continue;

    // all coincidences detIDs will have layer=1
    if (id.layer() != 1)
      continue;

    // find all corresponding ids with layer 2 and same roll that differs at most maxDeltaRoll_
    for (unsigned int roll = id.roll() - maxDeltaRoll_; roll <= id.roll() + maxDeltaRoll_; ++roll) {
      GEMDetId co_id(id.region(), id.ring(), id.station(), 2, id.chamber(), roll);

      auto co_pads_range = in_pads->get(co_id);

      // empty range = no possible coincidence pads
      if (co_pads_range.first == co_pads_range.second)
        continue;

      // now let's correlate the pads in two layers of this partition
      const auto& pads_range = (*det_range).second;
      for (auto p = pads_range.first; p != pads_range.second; ++p) {
        for (auto co_p = co_pads_range.first; co_p != co_pads_range.second; ++co_p) {
          // check the match in pad
          if ((unsigned)std::abs(p->pad() - co_p->pad()) > maxDeltaPad_)
            continue;

          // check the match in BX
          if ((unsigned)std::abs(p->bx() - co_p->bx()) > maxDeltaBX_)
            continue;

          // make a new coincidence pad digi
          gemCoPadV.push_back(GEMCoPadDigi(id.roll(), *p, *co_p));
        }
      }
    }
  }
  return gemCoPadV;
}

std::vector<GEMCoPadDigi> GEMCoPadProcessor::run(const GEMPadDigiClusterCollection* in_clusters) {
  std::unique_ptr<GEMPadDigiCollection> out_pads(new GEMPadDigiCollection());
  declusterize(in_clusters, *out_pads);
  return run(out_pads.get());
}

const std::vector<GEMCoPadDigi>& GEMCoPadProcessor::readoutCoPads() const { return gemCoPadV; }

void GEMCoPadProcessor::declusterize(const GEMPadDigiClusterCollection* in_clusters,
                                     GEMPadDigiCollection& out_pads) const {
  for (auto detUnitIt = in_clusters->begin(); detUnitIt != in_clusters->end(); ++detUnitIt) {
    const GEMDetId& id = (*detUnitIt).first;
    const auto& range = (*detUnitIt).second;
    for (auto digiIt = range.first; digiIt != range.second; ++digiIt) {
      for (const auto& p : digiIt->pads()) {
        out_pads.insertDigi(id, GEMPadDigi(p, digiIt->bx()));
      }
    }
  }
}
