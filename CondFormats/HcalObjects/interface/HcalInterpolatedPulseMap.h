#ifndef CondFormats_HcalObjects_HcalInterpolatedPulseMap_h_
#define CondFormats_HcalObjects_HcalInterpolatedPulseMap_h_

#include <map>
#include <string>

#include "boost/serialization/map.hpp"

#include "CondFormats/HcalObjects/interface/HcalInterpolatedPulse.h"

class HcalInterpolatedPulseMap {
public:
    void add(const std::string& label, const HcalInterpolatedPulse& pulse);
    inline void clear() { map_.clear(); }

    inline bool empty() const { return map_.empty(); }
    inline unsigned size() const { return map_.size(); }
    inline bool exists(const std::string& label) const {
        return !(map_.find(label) == map_.end());
    }
    const HcalInterpolatedPulse& get(const std::string& label) const;

    inline bool operator==(const HcalInterpolatedPulseMap& r) const { return map_ == r.map_; }
    inline bool operator!=(const HcalInterpolatedPulseMap& r) const { return !(*this == r); }

    void readFromTxt(const std::string& filename);

    // Precision of 0 means use default
    void dumpToTxt(const std::string& filename, unsigned precision = 0U) const;

private:
  // Return "true" if the line was parsed correctly
  // (even if nothing was added)
  bool addFromLine(const std::string& line);

  typedef std::map<std::string,HcalInterpolatedPulse> PulseMap;
  PulseMap map_;

  friend class boost::serialization::access;

  template <class Archive>
  inline void serialize(Archive& ar, unsigned /* version */) {
    ar & map_;  
  }
};

#endif // CondFormats_HcalObjects_HcalInterpolatedPulseMap_h_
