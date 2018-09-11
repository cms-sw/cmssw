#ifndef DataFormats_BTauReco_DeepBoostedJetFeatures_h
#define DataFormats_BTauReco_DeepBoostedJetFeatures_h

#include <string>
#include <vector>
#include <unordered_map>
#include "FWCore/Utilities/interface/Exception.h"

namespace btagbtvdeep {

class DeepBoostedJetFeatures {

public:

  bool empty() const {
    return is_empty_;
  }

  void add(const std::string& name){
    feature_map_[name];
  }

  void reserve(const std::string& name, unsigned capacity){
    feature_map_[name].reserve(capacity);
  }

  void fill(const std::string& name, float value){
    auto item = feature_map_.find(name);
    if (item != feature_map_.end()){
      item->second.push_back(value);
      is_empty_ = false;
    }else{
      throw cms::Exception("InvalidArgument") << "[DeepBoostedJetFeatures::fill()] Feature " << name << " has not been registered";
    }
  }

  void set(const std::string& name, const std::vector<float>& vec){
    feature_map_[name] = vec;
  }

  void check_consistency(const std::vector<std::string> &names) const {
    if (names.empty()) return;
    const auto ref_len = get(names.front()).size();
    for (unsigned i=1; i<names.size(); ++i){
      if (get(names[i]).size() != ref_len){
        throw cms::Exception("InvalidArgument") << "[DeepBoostedJetFeatures::check_consistency()] Inconsistent variable length "
            << get(names[i]).size() << " for " << names[i] << ", should be " << ref_len;
      }
    }
  }

  const std::vector<float>& get(const std::string& name) const {
    auto item = feature_map_.find(name);
    if (item != feature_map_.end()){
      return item->second;
    }else{
      throw cms::Exception("InvalidArgument") << "[DeepBoostedJetFeatures::get()] Feature " << name << " does not exist!";
    }
  }

private:
  bool is_empty_ = true;
  std::unordered_map<std::string, std::vector<float>> feature_map_;

};

}

#endif // DataFormats_BTauReco_DeepBoostedJetFeatures_h
