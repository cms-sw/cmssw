#ifndef StMeasurementDetSet_H
#define StMeasurementDetSet_H

#include <vector>
class TkStripMeasurementDet;
class TkStripMeasurementDet;
class TkPixelMeasurementDet;
class SiStripRecHitMatcher;
class StripClusterParameterEstimator;
class PixelClusterParameterEstimator;
class Phase2StripCPE;

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"
#include "DataFormats/Common/interface/Handle.h"

#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <atomic>
#include <unordered_map>

// #define VISTAT

#ifdef VISTAT
#include <iostream>
#define COUT std::cout
#else
#define COUT LogDebug("")
#endif

/* Struct of arrays supporting "members of Tk...MeasurementDet
 * implemented with vectors, to be optimized...
   ITEMS THAT DO NOT DEPEND ON THE EVENT
 */
class StMeasurementConditionSet {
public:
  enum QualityFlags {
    BadModules = 1,  // for everybody
    /* Strips: */ BadAPVFibers = 2,
    BadStrips = 4,
    MaskBad128StripBlocks = 8,
    /* Pixels: */ BadROCs = 2
  };

  struct BadStripCuts {
    BadStripCuts() : maxBad(9999), maxConsecutiveBad(9999) {}
    BadStripCuts(uint32_t iMaxBad, uint32_t iMaxConsecutiveBad)
        : maxBad(iMaxBad), maxConsecutiveBad(iMaxConsecutiveBad) {}
    uint16_t maxBad, maxConsecutiveBad;
  };

  struct BadStripBlock {
    short first;
    short last;
    BadStripBlock(const SiStripBadStrip::data& data) : first(data.firstStrip), last(data.firstStrip + data.range - 1) {}
  };

  StMeasurementConditionSet(const SiStripRecHitMatcher* matcher, const StripClusterParameterEstimator* cpe)
      : theMatcher(matcher), theCPE(cpe) {}

  void init(int size);

  const SiStripRecHitMatcher* matcher() const { return theMatcher; }
  const StripClusterParameterEstimator* stripCPE() const { return theCPE; }

  int nDet() const { return id_.size(); }
  unsigned int id(int i) const { return id_[i]; }
  unsigned char subId(int i) const { return subId_[i]; }

  int find(unsigned int jd, int i = 0) const { return std::lower_bound(id_.begin() + i, id_.end(), jd) - id_.begin(); }

  bool isActiveThisPeriod(int i) const { return activeThisPeriod_[i]; }

  /** \brief Turn on/off the module for reconstruction, for the full run or lumi (using info from DB, usually) */
  void setActive(int i, bool active) { activeThisPeriod_[i] = active; }

  int totalStrips(int i) const { return totalStrips_[i]; }

  void setMaskBad128StripBlocks(bool maskThem) { maskBad128StripBlocks_ = maskThem; }
  const BadStripCuts& badStripCuts(int i) const { return badStripCuts_[subId_[i]]; }

  bool maskBad128StripBlocks() const { return maskBad128StripBlocks_; }
  bool hasAny128StripBad(int i) const { return hasAny128StripBad_[i]; }

  /// note: index is 6*detector index + offset!
  bool bad128Strip(int offset) const { return bad128Strip_[offset]; }
  bool bad128Strip(int index, int strip) const { return bad128Strip_[nbad128 * index + (strip >> 7)]; }

  std::vector<BadStripBlock>& getBadStripBlocks(int i) { return badStripBlocks_[i]; }
  std::vector<BadStripBlock> const& badStripBlocks(int i) const { return badStripBlocks_[i]; }

  bool isMasked(int i, const SiStripCluster& cluster) const {
    int offset = nbad128 * i;
    if (bad128Strip_[offset + (cluster.firstStrip() >> 7)]) {
      if (bad128Strip_[offset + ((cluster.firstStrip() + cluster.amplitudes().size()) >> 7)] ||
          bad128Strip_[offset + (static_cast<int32_t>(cluster.barycenter() - 0.499999) >> 7)]) {
        return true;
      }
    } else {
      if (bad128Strip_[offset + ((cluster.firstStrip() + cluster.amplitudes().size()) >> 7)] &&
          bad128Strip_[offset + (static_cast<int32_t>(cluster.barycenter() - 0.499999) >> 7)]) {
        return true;
      }
    }
    return false;
  }

  void set128StripStatus(int i, bool good, int idx = -1);

private:
  friend class MeasurementTrackerImpl;

  // globals
  const SiStripRecHitMatcher* theMatcher;
  const StripClusterParameterEstimator* theCPE;

  bool maskBad128StripBlocks_;
  BadStripCuts badStripCuts_[4];

  // members of TkStripMeasurementDet
  std::vector<unsigned int> id_;
  std::vector<unsigned char> subId_;

  std::vector<int> totalStrips_;

  static const int nbad128 = 6;
  std::vector<bool> bad128Strip_;
  std::vector<bool> hasAny128StripBad_;

  std::vector<std::vector<BadStripBlock>> badStripBlocks_;

  std::vector<bool> activeThisPeriod_;
};

class StMeasurementDetSet {
public:
  typedef edmNew::DetSet<SiStripCluster> StripDetset;
  typedef StripDetset::const_iterator new_const_iterator;

  StMeasurementDetSet(const StMeasurementConditionSet& cond)
      : conditionSet_(&cond),
        activeThisEvent_(cond.nDet(), true),
        detSet_(cond.nDet()),
        detIndex_(cond.nDet(), -1),
        theRawInactiveStripDetIds_(),
        stripDefined_(0),
        stripUpdated_(0),
        stripRegions_(0) {}

  ~StMeasurementDetSet() { printStat(); }

  const StMeasurementConditionSet& conditions() const { return *conditionSet_; }

  void update(int i, const StripDetset& detSet) {
    detSet_[i].detSet_ = detSet;
    detSet_[i].empty_ = false;
  }

  void update(int i, int j) {
    assert(j >= 0);
    assert(detSet_[i].empty_);
    assert(detSet_[i].ready_);
    detIndex_[i] = j;
    detSet_[i].empty_ = false;
    incReady();
  }

  int size() const { return conditions().nDet(); }
  int nDet() const { return size(); }
  unsigned int id(int i) const { return conditions().id(i); }
  int find(unsigned int jd, int i = 0) const { return conditions().find(jd, i); }

  bool empty(int i) const { return detSet_[i].empty_; }
  bool isActive(int i) const { return activeThisEvent_[i] && conditions().isActiveThisPeriod(i); }

  void setEmpty(int i) {
    detSet_[i].empty_ = true;
    activeThisEvent_[i] = true;
  }
  void setUpdated(int i) { stripUpdated_[i] = true; }

  void setEmpty() {
    printStat();
    for (auto& d : detSet_) {
      d.empty_ = true;
      d.ready_ = true;
    }
    std::fill(detIndex_.begin(), detIndex_.end(), -1);
    std::fill(activeThisEvent_.begin(), activeThisEvent_.end(), true);
    incTot(size());
  }

  /** \brief Turn on/off the module for reconstruction for one events.
      This per-event flag is cleared by any call to 'update' or 'setEmpty'  */
  void setActiveThisEvent(int i, bool active) {
    activeThisEvent_[i] = active;
    if (!active)
      detSet_[i].empty_ = true;
  }

  edm::Handle<edmNew::DetSetVector<SiStripCluster>>& handle() { return handle_; }
  const edm::Handle<edmNew::DetSetVector<SiStripCluster>>& handle() const { return handle_; }
  // StripDetset & detSet(int i) { return detSet_[i]; }
  const StripDetset& detSet(int i) const {
    if (detSet_[i].ready_)
      getDetSet(i);
    return detSet_[i].detSet_;
  }

  //// ------- pieces for on-demand unpacking --------
  std::vector<uint32_t>& rawInactiveStripDetIds() { return theRawInactiveStripDetIds_; }
  const std::vector<uint32_t>& rawInactiveStripDetIds() const { return theRawInactiveStripDetIds_; }

  void resetOnDemandStrips() {
    std::fill(stripDefined_.begin(), stripDefined_.end(), false);
    std::fill(stripUpdated_.begin(), stripUpdated_.end(), false);
  }
  const bool stripDefined(int i) const { return stripDefined_[i]; }
  const bool stripUpdated(int i) const { return stripUpdated_[i]; }
  void defineStrip(int i, std::pair<unsigned int, unsigned int> range) {
    stripDefined_[i] = true;
    stripUpdated_[i] = false;
    stripRegions_[i] = range;
  }

private:
  void getDetSet(int i) const {
    const auto& det = detSet_[i];
    if (detIndex_[i] >= 0) {
      det.detSet_ = StripDetset(*handle_, handle_->item(detIndex_[i]), true);
      det.empty_ = false;  // better be false already
      incAct();
    } else {  // we should not be here
      det.detSet_ = StripDetset();
      det.empty_ = true;
    }
    det.ready_ = false;
    incSet();
  }

  friend class MeasurementTrackerImpl;

  const StMeasurementConditionSet* conditionSet_;

  // Globals, per-event
  edm::Handle<edmNew::DetSetVector<SiStripCluster>> handle_;

  // Helper struct to define only the vector elements as mutable and
  // to have a vector of atomics without an explicit loop over
  // elements to set their values
  struct DetSetHelper {
    mutable std::atomic<bool> empty_ = true;
    mutable std::atomic<bool> ready_ = true;  // to be cleaned
    // only thread-safe non-const member functions are called from a const function
    CMS_THREAD_SAFE mutable StripDetset detSet_;
  };

  std::vector<bool> activeThisEvent_;

  // full reco
  std::vector<DetSetHelper> detSet_;
  std::vector<int> detIndex_;

  // note: not aligned to the index
  std::vector<uint32_t> theRawInactiveStripDetIds_;
  // keyed on si-strip index
  std::vector<bool> stripDefined_, stripUpdated_;
  std::vector<std::pair<unsigned int, unsigned int>> stripRegions_;
  // keyed on glued
  // std::vector<bool> gluedUpdated_;

#ifdef VISTAT
  struct Stat {
    int totDet = 0;    // all dets
    int detReady = 0;  // dets "updated"
    int detSet = 0;    // det actually set not empty
    int detAct = 0;    // det actually set with content
  };

  mutable Stat stat;
  void zeroStat() const { stat = Stat(); }
  void incTot(int n) const { stat.totDet = n; }
  void incReady() const { stat.detReady++; }
  void incSet() const { stat.detSet++; }
  void incAct() const { stat.detAct++; }
  void printStat() const {
    COUT << "VI detsets " << stat.totDet << ',' << stat.detReady << ',' << stat.detSet << ',' << stat.detAct
         << std::endl;
  }

#else
  static void zeroStat() {}
  static void incTot(int) {}
  static void incReady() {}
  static void incSet() {}
  static void incAct() {}
  static void printStat() {}
#endif
};

class PxMeasurementConditionSet {
public:
  PxMeasurementConditionSet(const PixelClusterParameterEstimator* cpe) : theCPE(cpe) {}

  void init(int size);

  int nDet() const { return id_.size(); }
  unsigned int id(int i) const { return id_[i]; }
  int find(unsigned int jd, int i = 0) const { return std::lower_bound(id_.begin() + i, id_.end(), jd) - id_.begin(); }

  const PixelClusterParameterEstimator* pixelCPE() const { return theCPE; }
  bool isActiveThisPeriod(int i) const { return activeThisPeriod_[i]; }

  /** \brief Turn on/off the module for reconstruction, for the full run or lumi (using info from DB, usually).
      This also resets the 'setActiveThisEvent' to true */
  void setActive(int i, bool active) { activeThisPeriod_[i] = active; }

private:
  friend class MeasurementTrackerImpl;

  // Globals (not-per-event)
  const PixelClusterParameterEstimator* theCPE;

  // Locals, per-event
  std::vector<unsigned int> id_;
  std::vector<bool> activeThisPeriod_;
};

class PxMeasurementDetSet {
public:
  typedef edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> SiPixelClusterRef;
  typedef edmNew::DetSet<SiPixelCluster> PixelDetSet;
  typedef std::vector<std::pair<LocalPoint, LocalPoint>> BadFEDChannelPositions;

  PxMeasurementDetSet(const PxMeasurementConditionSet& cond)
      : conditionSet_(&cond), detSet_(cond.nDet()), empty_(cond.nDet(), true), activeThisEvent_(cond.nDet(), true) {}

  const PxMeasurementConditionSet& conditions() const { return *conditionSet_; }

  int size() const { return conditions().nDet(); }
  int nDet() const { return size(); }
  unsigned int id(int i) const { return conditions().id(i); }
  int find(unsigned int jd, int i = 0) const { return conditions().find(jd, i); }

  void update(int i, const PixelDetSet& detSet) {
    detSet_[i] = detSet;
    empty_[i] = false;
  }

  bool empty(int i) const { return empty_[i]; }
  bool isActive(int i) const { return activeThisEvent_[i] && conditions().isActiveThisPeriod(i); }

  void setEmpty(int i) {
    empty_[i] = true;
    activeThisEvent_[i] = true;
    auto found = badFEDChannelPositionsSet_.find(i);
    if (found != badFEDChannelPositionsSet_.end()) {
      badFEDChannelPositionsSet_.erase(found);
    }
  }

  void setEmpty() {
    std::fill(empty_.begin(), empty_.end(), true);
    std::fill(activeThisEvent_.begin(), activeThisEvent_.end(), true);
    badFEDChannelPositionsSet_.clear();
  }
  void setActiveThisEvent(bool active) { std::fill(activeThisEvent_.begin(), activeThisEvent_.end(), active); }

  const BadFEDChannelPositions* getBadFEDChannelPositions(int i) const {
    auto found = badFEDChannelPositionsSet_.find(i);
    if (found == badFEDChannelPositionsSet_.end())
      return nullptr;
    return &(found->second);
  }
  void addBadFEDChannelPositions(int i, BadFEDChannelPositions& positions) {
    auto found = badFEDChannelPositionsSet_.find(i);
    if (found == badFEDChannelPositionsSet_.end()) {
      badFEDChannelPositionsSet_.emplace(i, positions);
    } else {
      found->second.insert(found->second.end(), positions.begin(), positions.end());
    }
  }

  /** \brief Turn on/off the module for reconstruction for one events.
      This per-event flag is cleared by any call to 'update' or 'setEmpty'  */
  void setActiveThisEvent(int i, bool active) {
    activeThisEvent_[i] = active;
    if (!active)
      empty_[i] = true;
  }
  const edm::Handle<edmNew::DetSetVector<SiPixelCluster>>& handle() const { return handle_; }
  edm::Handle<edmNew::DetSetVector<SiPixelCluster>>& handle() { return handle_; }
  const PixelDetSet& detSet(int i) const { return detSet_[i]; }

private:
  friend class MeasurementTrackerImpl;

  const PxMeasurementConditionSet* conditionSet_;

  // Globals, per-event
  edm::Handle<edmNew::DetSetVector<SiPixelCluster>> handle_;

  // Locals, per-event
  std::vector<PixelDetSet> detSet_;
  std::vector<bool> empty_;
  std::vector<bool> activeThisEvent_;
  std::unordered_map<int, BadFEDChannelPositions> badFEDChannelPositionsSet_;
};

//FIXME:just temporary solution for phase2 OT that works!
class Phase2OTMeasurementConditionSet {
public:
  Phase2OTMeasurementConditionSet(const ClusterParameterEstimator<Phase2TrackerCluster1D>* cpe) : theCPE(cpe) {}

  void init(int size);

  int nDet() const { return id_.size(); }
  unsigned int id(int i) const { return id_[i]; }
  int find(unsigned int jd, int i = 0) const { return std::lower_bound(id_.begin() + i, id_.end(), jd) - id_.begin(); }

  const ClusterParameterEstimator<Phase2TrackerCluster1D>* cpe() const { return theCPE; }
  bool isActiveThisPeriod(int i) const { return activeThisPeriod_[i]; }

  /** \brief Turn on/off the module for reconstruction, for the full run or lumi (using info from DB, usually).
 *       This also resets the 'setActiveThisEvent' to true */
  void setActive(int i, bool active) { activeThisPeriod_[i] = active; }

private:
  friend class MeasurementTrackerImpl;

  // Globals (not-per-event)
  const ClusterParameterEstimator<Phase2TrackerCluster1D>* theCPE;

  // Locals, per-event
  std::vector<unsigned int> id_;
  std::vector<bool> activeThisPeriod_;
};

class Phase2OTMeasurementDetSet {
public:
  typedef edm::Ref<edmNew::DetSetVector<Phase2TrackerCluster1D>, Phase2TrackerCluster1D> Phase2TrackerCluster1DRef;
  typedef edmNew::DetSet<Phase2TrackerCluster1D> Phase2DetSet;

  Phase2OTMeasurementDetSet(const Phase2OTMeasurementConditionSet& cond)
      : conditionSet_(&cond), detSet_(cond.nDet()), empty_(cond.nDet(), true), activeThisEvent_(cond.nDet(), true) {}

  const Phase2OTMeasurementConditionSet& conditions() const { return *conditionSet_; }

  int size() const { return conditions().nDet(); }
  int nDet() const { return size(); }
  unsigned int id(int i) const { return conditions().id(i); }
  int find(unsigned int jd, int i = 0) const { return conditions().find(jd, i); }

  void update(int i, const Phase2DetSet& detSet) {
    detSet_[i] = detSet;
    empty_[i] = false;
  }

  bool empty(int i) const { return empty_[i]; }
  bool isActive(int i) const { return activeThisEvent_[i] && conditions().isActiveThisPeriod(i); }

  void setEmpty(int i) {
    empty_[i] = true;
    activeThisEvent_[i] = true;
  }

  void setEmpty() {
    std::fill(empty_.begin(), empty_.end(), true);
    std::fill(activeThisEvent_.begin(), activeThisEvent_.end(), true);
  }
  void setActiveThisEvent(bool active) { std::fill(activeThisEvent_.begin(), activeThisEvent_.end(), active); }
  void setActiveThisEvent(int i, bool active) {
    activeThisEvent_[i] = active;
    if (!active)
      empty_[i] = true;
  }
  const edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>>& handle() const { return handle_; }
  edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>>& handle() { return handle_; }
  const Phase2DetSet& detSet(int i) const { return detSet_[i]; }

private:
  friend class MeasurementTrackerImpl;

  const Phase2OTMeasurementConditionSet* conditionSet_;

  //Globals, per-event
  edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> handle_;

  // Locals, per-event
  std::vector<Phase2DetSet> detSet_;
  std::vector<bool> empty_;
  std::vector<bool> activeThisEvent_;
};

#endif  // StMeasurementDetSet_H
