#ifndef PhysicsTools_PatAlgos_XGBooster_h
#define PhysicsTools_PatAlgos_XGBooster_h

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <xgboost/c_api.h>

namespace pat {
  class XGBooster {
  public:
    XGBooster(std::string model_file);
    XGBooster(std::string model_file, std::string model_features);

    /// Features need to be entered in the order they are used
    /// in the model
    void addFeature(std::string name);

    /// Reset feature values
    void reset();

    void set(std::string name, float value);

    float predict();

  private:
    std::vector<float> features_;
    std::map<std::string, unsigned int> feature_name_to_index_;
    BoosterHandle booster_;
  };
}  // namespace pat

#endif
