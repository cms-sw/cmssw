#include "PhysicsTools/PatAlgos/interface/XGBooster.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>

using namespace pat;

std::vector<std::string> read_features(const std::string& content) {
  std::vector<std::string> result;

  std::istringstream stream(content);
  char ch;

  // Expect opening '['
  stream >> ch;
  if (ch != '[') {
    throw std::runtime_error("Expected '[' at the beginning of the JSON array!");
  }

  while (stream) {
    stream >> ch;

    if (ch == ']') {
      break;
    } else if (ch == ',') {
      continue;
    } else if (ch == '"') {
      std::string feature;
      std::getline(stream, feature, '"');
      result.push_back(feature);
    } else {
      throw std::runtime_error("Unexpected character in the JSON array!");
    }
  }

  return result;
}

XGBooster::XGBooster(std::string model_file) {
  int status = XGBoosterCreate(nullptr, 0, &booster_);
  if (status != 0)
    throw std::runtime_error("Failed to create XGBooster");
  status = XGBoosterLoadModel(booster_, model_file.c_str());
  if (status != 0)
    throw std::runtime_error("Failed to load XGBoost model");
  XGBoosterSetParam(booster_, "nthread", "1");
}

XGBooster::XGBooster(std::string model_file, std::string model_features) : XGBooster(model_file) {
  std::ifstream file(model_features);
  if (!file.is_open())
    throw std::runtime_error("Failed to open file: " + model_features);

  std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();

  std::vector<std::string> features = read_features(content);

  for (const auto& feature : features) {
    addFeature(feature);
  }
}

void XGBooster::reset() { std::fill(features_.begin(), features_.end(), std::nan("")); }

void XGBooster::addFeature(std::string name) {
  features_.push_back(0);
  feature_name_to_index_[name] = features_.size() - 1;
}

void XGBooster::set(std::string name, float value) { features_.at(feature_name_to_index_[name]) = value; }

float XGBooster::predict() {
  float result(-999.);

  // check if all feature values are set properly
  for (unsigned int i = 0; i < features_.size(); ++i)
    if (std::isnan(features_.at(i))) {
      std::string feature_name;
      for (const auto& pair : feature_name_to_index_) {
        if (pair.second == i) {
          feature_name = pair.first;
          break;
        }
      }
      throw std::runtime_error("Feature is not set: " + feature_name);
    }

  DMatrixHandle dvalues;
  XGDMatrixCreateFromMat(&features_[0], 1, features_.size(), 9e99, &dvalues);

  bst_ulong out_len = 0;
  const float* score = nullptr;

  // config json
  const char* json = R"({
    "type": 0,
    "training": false,
    "iteration_begin": 0,
    "iteration_end": 0,
    "strict_shape": false
   })";

  // Shape of output prediction
  bst_ulong const* out_shape = nullptr;

  auto ret = XGBoosterPredictFromDMatrix(booster_, dvalues, json, &out_shape, &out_len, &score);

  XGDMatrixFree(dvalues);

  if (ret == 0) {
    assert(out_len == 1 && "Unexpected prediction format");
    result = score[0];
  }

  reset();

  return result;
}
