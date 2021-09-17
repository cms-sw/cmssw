#ifndef CondFormats_HcalObjects_OOTPileupCorrectionColl_h
#define CondFormats_HcalObjects_OOTPileupCorrectionColl_h

#include <string>

#include "boost/serialization/shared_ptr.hpp"
#include "boost/serialization/map.hpp"

#include "CondFormats/HcalObjects/interface/AbsOOTPileupCorrection.h"

class OOTPileupCorrectionColl {
public:
  inline void add(const std::string& name, const std::string& category, std::shared_ptr<AbsOOTPileupCorrection> ptr) {
    data_[category][name] = ptr;
  }

  inline void clear() { data_.clear(); }

  inline bool empty() const { return data_.empty(); }

  std::shared_ptr<AbsOOTPileupCorrection> get(const std::string& name, const std::string& category) const;

  bool exists(const std::string& name, const std::string& category) const;

  bool operator==(const OOTPileupCorrectionColl& r) const;

  inline bool operator!=(const OOTPileupCorrectionColl& r) const { return !(*this == r); }

private:
  typedef std::map<std::string, std::shared_ptr<AbsOOTPileupCorrection> > PtrMap;
  typedef std::map<std::string, PtrMap> DataMap;
  DataMap data_;

  friend class boost::serialization::access;

  template <class Archive>
  inline void serialize(Archive& ar, unsigned /* version */) {
    ar& data_;
  }
};

#endif  // CondFormats_HcalObjects_OOTPileupCorrectionColl_h
