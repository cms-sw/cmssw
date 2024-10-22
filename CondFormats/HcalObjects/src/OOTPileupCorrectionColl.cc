#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/HcalObjects/interface/OOTPileupCorrectionColl.h"

bool OOTPileupCorrectionColl::exists(const std::string& name, const std::string& category) const {
  DataMap::const_iterator dit = data_.find(category);
  if (dit == data_.end())
    return false;
  else
    return !(dit->second.find(name) == dit->second.end());
}

std::shared_ptr<AbsOOTPileupCorrection> OOTPileupCorrectionColl::get(const std::string& name,
                                                                     const std::string& category) const {
  DataMap::const_iterator dit = data_.find(category);
  if (dit == data_.end())
    throw cms::Exception("In OOTPileupCorrectionColl::get: unknown category");
  PtrMap::const_iterator pit = dit->second.find(name);
  if (pit == dit->second.end())
    throw cms::Exception("In OOTPileupCorrectionColl::get: unknown name");
  return pit->second;
}

bool OOTPileupCorrectionColl::operator==(const OOTPileupCorrectionColl& r) const {
  if (data_.size() != r.data_.size())
    return false;
  DataMap::const_iterator dit = data_.begin();
  const DataMap::const_iterator end = data_.end();
  DataMap::const_iterator rit = r.data_.begin();
  for (; dit != end; ++dit, ++rit) {
    if (dit->first != rit->first)
      return false;
    if (dit->second.size() != rit->second.size())
      return false;
    PtrMap::const_iterator pit = dit->second.begin();
    const PtrMap::const_iterator pend = dit->second.end();
    PtrMap::const_iterator rpit = rit->second.begin();
    for (; pit != pend; ++pit, ++rpit) {
      if (pit->first != rpit->first)
        return false;
      if (*(pit->second) != *(rpit->second))
        return false;
    }
  }
  return true;
}
