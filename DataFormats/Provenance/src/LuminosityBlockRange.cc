#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
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

  std::ostream& operator<<(std::ostream& oStream, LuminosityBlockRange const& r) {
    oStream << "'" << r.startRun() << ":" << r.startLumi() << "-"
                   << r.endRun()   << ":" << r.endLumi()   << "'" ;
    return oStream;
  }

  bool LuminosityBlockRange::contains(LuminosityBlockID const& test) const {
    if (test >= startLumiID_ && test <= endLumiID_) {
      return true;
    }
    return false;
  }

}
