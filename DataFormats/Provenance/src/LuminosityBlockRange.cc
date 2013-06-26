#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include <cassert>
#include <ostream>
//#include <limits>



namespace edm {

//   static unsigned int const shift = 8 * sizeof(unsigned int);
//
//   LuminosityBlockID::LuminosityBlockID(boost::uint64_t id) :
//    run_(static_cast<RunNumber_t>(id >> shift)),
//    luminosityBlock_(static_cast<LuminosityBlockNumber_t>(std::numeric_limits<unsigned int>::max() & id))
//   {
//   }
//
//   boost::uint64_t
//   LuminosityBlockID::value() const {
//    boost::uint64_t id = run_;
//    id = id << shift;
//    id += luminosityBlock_;
//    return id;
//   }


  LuminosityBlockRange::LuminosityBlockRange() :
    // Special cases since 0 means maximum
    startLumiID_(0, LuminosityBlockID::maxLuminosityBlockNumber()),
    endLumiID_(0, LuminosityBlockID::maxLuminosityBlockNumber()){
  }

  LuminosityBlockRange::LuminosityBlockRange(RunNumber_t startRun, LuminosityBlockNumber_t startLuminosityBlock,
                                             RunNumber_t endRun,   LuminosityBlockNumber_t endLuminosityBlock) :
    // Special cases since 0 means maximum
    startLumiID_(startRun, startLuminosityBlock != 0 ? startLuminosityBlock : LuminosityBlockID::maxLuminosityBlockNumber()),
    endLumiID_(endRun, endLuminosityBlock !=0 ? endLuminosityBlock : LuminosityBlockID::maxLuminosityBlockNumber()) {
  }

  LuminosityBlockRange::LuminosityBlockRange(LuminosityBlockID const& begin, LuminosityBlockID const& end) :
    startLumiID_(begin),
    endLumiID_(end) {
  }

  std::ostream& operator<<(std::ostream& oStream, LuminosityBlockRange const& r) {
    oStream << "'" << r.startRun() << ":" << r.startLumi() << "-"
                   << r.endRun()   << ":" << r.endLumi()   << "'" ;
    return oStream;
  }

  bool contains(LuminosityBlockRange const& lh, LuminosityBlockID const& rh) {
    if (rh >= lh.startLumiID() && rh <= lh.endLumiID()) {
      return true;
    }
    return false;
  }

  bool contains(LuminosityBlockRange const& lh, LuminosityBlockRange const& rh) {
    if (contains(lh,rh.startLumiID()) && contains(lh,rh.endLumiID())) {
      return true;
    }
    return false;
  }

  bool overlaps(LuminosityBlockRange const& lh, LuminosityBlockRange const& rh) {
    return !distinct(lh, rh);
  }

  bool lessThan(LuminosityBlockRange const& lh, LuminosityBlockRange const& rh) {
    return lh.endLumiID() < rh.startLumiID();
  }

  bool distinct(LuminosityBlockRange const& lh, LuminosityBlockRange const& rh) {
    return lessThan(lh, rh) || lessThan(rh, lh);
  }

  bool merge(LuminosityBlockRange& lh, LuminosityBlockRange& rh) {
    if (overlaps(lh, rh)) {
      LuminosityBlockID begin = min(lh.startLumiID(), rh.startLumiID());
      LuminosityBlockID end = max(lh.endLumiID(), rh.endLumiID());
      rh = lh = LuminosityBlockRange(begin, end);
      return true;
    }
    return false;
  }

  namespace {
    bool sortByStartLuminosityBlockID(LuminosityBlockRange const& lh, LuminosityBlockRange const& rh) {
      assert((lh.startLumi() == 0) == (rh.startLumi() == 0));
      return lh.startLumiID() < rh.startLumiID();
    }
  }

  std::vector<LuminosityBlockRange>&
  sortAndRemoveOverlaps(std::vector<LuminosityBlockRange>& lumiRange) {
    if (lumiRange.size() <= 1U) return lumiRange;
    sort_all(lumiRange, sortByStartLuminosityBlockID);
    for (std::vector<LuminosityBlockRange>::iterator i = lumiRange.begin() + 1, e = lumiRange.end();
        i != e; ++i) {
      std::vector<LuminosityBlockRange>::iterator iprev = i - 1;
      if (merge(*iprev, *i)) {
        i = lumiRange.erase(iprev);
        e = lumiRange.end();
      }
    }
    return lumiRange;
  }
}
