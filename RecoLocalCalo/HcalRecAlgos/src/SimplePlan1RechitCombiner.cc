#include <cassert>
#include <algorithm>

#include "DataFormats/HcalRecHit/interface/HcalSpecialTimes.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHitAuxSetter.h"
#include "DataFormats/HcalRecHit/interface/CaloRecHitAuxSetter.h"

#include "DataFormats/METReco/interface/HcalPhase1FlagLabels.h"

#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/SimplePlan1RechitCombiner.h"

SimplePlan1RechitCombiner::SimplePlan1RechitCombiner() : topo_(nullptr) {}

void SimplePlan1RechitCombiner::setTopo(const HcalTopology* topo) { topo_ = topo; }

void SimplePlan1RechitCombiner::clear() { rechitMap_.clear(); }

void SimplePlan1RechitCombiner::add(const HBHERecHit& rh) {
  if (!CaloRecHitAuxSetter::getBit(rh.auxPhase1(), HBHERecHitAuxSetter::OFF_DROPPED))
    rechitMap_.push_back(MapItem(mapRechit(rh.id()), &rh));
}

void SimplePlan1RechitCombiner::combine(HBHERecHitCollection* toFill) {
  if (!rechitMap_.empty()) {
    std::sort(rechitMap_.begin(), rechitMap_.end());

    HcalDetId oldId(rechitMap_[0].first);
    ptrbuf_.clear();
    ptrbuf_.push_back(rechitMap_[0].second);

    const std::size_t nInput = rechitMap_.size();
    for (std::size_t i = 1; i < nInput; ++i) {
      if (rechitMap_[i].first != oldId) {
        const HBHERecHit& rh = makeRechit(oldId, ptrbuf_);
        if (rh.id().rawId())
          toFill->push_back(rh);
        oldId = rechitMap_[i].first;
        ptrbuf_.clear();
      }
      ptrbuf_.push_back(rechitMap_[i].second);
    }

    const HBHERecHit& rh = makeRechit(oldId, ptrbuf_);
    if (rh.id().rawId())
      toFill->push_back(rh);
  }
}

HcalDetId SimplePlan1RechitCombiner::mapRechit(HcalDetId from) const { return topo_->mergedDepthDetId(from); }

HBHERecHit SimplePlan1RechitCombiner::makeRechit(const HcalDetId idToMake,
                                                 const std::vector<const HBHERecHit*>& rechits) const {
  constexpr unsigned MAXLEN = 8U;  // Should be >= max # of Phase 1 HCAL depths
  constexpr float TIME_IF_NO_ENERGY = -999.f;

  const unsigned nRecHits = rechits.size();
  assert(nRecHits);
  assert(nRecHits <= MAXLEN);

  // Combine energies, times, and fit chi-square
  double energy = 0.0, eraw = 0.0, eaux = 0.0, chisq = 0.0;
  FPair times[MAXLEN], adctimes[MAXLEN];
  unsigned nADCTimes = 0;

  for (unsigned i = 0; i < nRecHits; ++i) {
    const HBHERecHit& rh(*rechits[i]);
    const float e = rh.energy();
    energy += e;
    eraw += rh.eraw();
    eaux += rh.eaux();
    chisq += rh.chi2();
    times[i].first = rh.time();
    times[i].second = e;

    const float tADC = rh.timeFalling();
    if (!HcalSpecialTimes::isSpecial(tADC)) {
      adctimes[nADCTimes].first = tADC;
      adctimes[nADCTimes].second = e;
      ++nADCTimes;
    }
  }

  HBHERecHit rh(idToMake,
                energy,
                energyWeightedAverage(times, nRecHits, TIME_IF_NO_ENERGY),
                energyWeightedAverage(adctimes, nADCTimes, HcalSpecialTimes::UNKNOWN_T_NOTDC));
  rh.setRawEnergy(eraw);
  rh.setAuxEnergy(eaux);
  rh.setChiSquared(chisq);

  // Combine the auxiliary information
  combineAuxInfo(rechits, &rh);

  return rh;
}

void SimplePlan1RechitCombiner::combineAuxInfo(const std::vector<const HBHERecHit*>& rechits, HBHERecHit* rh) const {
  using namespace CaloRecHitAuxSetter;
  using namespace HcalPhase1FlagLabels;

  // The number of rechits should be not larger than the
  // number of half-bytes in a 32-bit word
  constexpr unsigned MAXLEN = 8U;

  const unsigned nRecHits = rechits.size();
  assert(nRecHits);
  assert(nRecHits <= MAXLEN);

  uint32_t flags = 0, auxPhase1 = 0;
  unsigned tripleFitCount = 0;
  unsigned soiVote[HBHERecHitAuxSetter::MASK_SOI + 1U] = {0};
  unsigned capidVote[HBHERecHitAuxSetter::MASK_CAPID + 1U] = {0};

  // Combine various status bits
  for (unsigned i = 0; i < nRecHits; ++i) {
    const HBHERecHit& rh(*rechits[i]);
    const uint32_t rhflags = rh.flags();
    const uint32_t rhAuxPhase1 = rh.auxPhase1();

    orBit(&auxPhase1, HBHERecHitAuxSetter::OFF_LINK_ERR, getBit(rhAuxPhase1, HBHERecHitAuxSetter::OFF_LINK_ERR));
    orBit(&auxPhase1, HBHERecHitAuxSetter::OFF_CAPID_ERR, getBit(rhAuxPhase1, HBHERecHitAuxSetter::OFF_CAPID_ERR));

    const unsigned soi = getField(rhAuxPhase1, HBHERecHitAuxSetter::MASK_SOI, HBHERecHitAuxSetter::OFF_SOI);
    soiVote[soi]++;

    const unsigned capid = getField(rhAuxPhase1, HBHERecHitAuxSetter::MASK_CAPID, HBHERecHitAuxSetter::OFF_CAPID);
    capidVote[capid]++;

    if (getBit(rhflags, HBHEStatusFlag::HBHEPulseFitBit))
      ++tripleFitCount;

    // Status flags are simply ORed for now. Might want
    // to rethink this in the future.
    flags |= rhflags;
  }

  unsigned* pmaxsoi = std::max_element(soiVote, soiVote + sizeof(soiVote) / sizeof(soiVote[0]));
  const unsigned soi = std::distance(&soiVote[0], pmaxsoi);
  setField(&auxPhase1, HBHERecHitAuxSetter::MASK_SOI, HBHERecHitAuxSetter::OFF_SOI, soi);

  unsigned* pmaxcapid = std::max_element(capidVote, capidVote + sizeof(capidVote) / sizeof(capidVote[0]));
  const unsigned capid = std::distance(&capidVote[0], pmaxcapid);
  setField(&auxPhase1, HBHERecHitAuxSetter::MASK_CAPID, HBHERecHitAuxSetter::OFF_CAPID, capid);

  // A number that can be later used to calculate chi-square NDoF
  setField(&auxPhase1, HBHERecHitAuxSetter::MASK_NSAMPLES, HBHERecHitAuxSetter::OFF_ADC, tripleFitCount);

  // How many rechits were combined?
  setField(&auxPhase1, HBHERecHitAuxSetter::MASK_NSAMPLES, HBHERecHitAuxSetter::OFF_NSAMPLES, nRecHits);

  // Should combine QIE11 data only
  setBit(&auxPhase1, HBHERecHitAuxSetter::OFF_TDC_TIME, true);

  // Indicate that this rechit is combined
  setBit(&auxPhase1, HBHERecHitAuxSetter::OFF_COMBINED, true);

  // Copy the aux words into the rechit
  rh->setFlags(flags);
  rh->setAuxPhase1(auxPhase1);

  // Sort the depth values of the combined rechits
  // in the increasing order
  unsigned depthValues[MAXLEN];
  for (unsigned i = 0; i < nRecHits; ++i)
    depthValues[i] = rechits[i]->id().depth();
  if (nRecHits > 1U)
    std::sort(depthValues, depthValues + nRecHits);

  // Pack the information about the depth of the rechits
  // that we are combining into the "auxHBHE" word
  uint32_t auxHBHE = 0;
  for (unsigned i = 0; i < nRecHits; ++i)
    setField(&auxHBHE, 0xf, i * 4, depthValues[i]);
  rh->setAuxHBHE(auxHBHE);
}
