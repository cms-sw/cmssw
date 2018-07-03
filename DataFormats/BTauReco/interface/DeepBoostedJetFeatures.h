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

  void fill(const std::string& name, float value){
    try {
      feature_map_.at(name).push_back(value);
      is_empty_ = false;
    }catch (const std::out_of_range& e) {
      throw cms::Exception("InvalidArgument") << "[DeepBoostedJetFeatures::fill()] Feature " << name << " has not been registered";
    }
  }

  void set(const std::string& name, const std::vector<float>& vec){
    feature_map_[name] = vec;
  }

  const std::vector<float>& get(const std::string& name) const {
    try {
      return feature_map_.at(name);
    }catch (const std::out_of_range& e) {
      throw cms::Exception("InvalidArgument") << "[DeepBoostedJetFeatures::get()] Feature " << name << " does not exist!";
    }
  }

private:
  bool is_empty_ = true;
  std::unordered_map<std::string, std::vector<float>> feature_map_;

};

}

#endif // DataFormats_BTauReco_DeepBoostedJetFeatures_h
