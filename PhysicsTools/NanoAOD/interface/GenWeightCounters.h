#ifndef GENWEIGHTSCOUNTERS_h
#define GENWEIGHTSCOUNTERS_h
#include <vector>
#include <map>
#include <string>
namespace genCounter {

  void mergeSumVectors(std::vector<long double>& v1, std::vector<long double> const& v2) {
    if (v1.empty() && !v2.empty())
      v1.resize(v2.size(), 0);
    if (!v2.empty())
      for (unsigned int i = 0, n = v1.size(); i < n; ++i)
        v1[i] += v2[i];
  }

  ///  ---- Cache object for running sums of weights ----
  class Counter {
  public:
    void clear() {
      num_ = 0;
      sumw_ = 0;
      sumw2_ = 0;
      weightSumMap_.clear();
    }

    // inc only the gen counters
    void incGenOnly(double w) {
      num_++;
      sumw_ += w;
      sumw2_ += (w * w);
    }

    void incLHE(double w0, const std::vector<double>& weightV, const std::string& wName) {
      //if new type of weight, create a map element
      if (weightSumMap_.find(wName) == weightSumMap_.end()) {
        std::vector<long double> temp;
        weightSumMap_.insert({wName, temp});
      }
      if (!weightV.empty()) {
        if (weightSumMap_[wName].empty())
          weightSumMap_[wName].resize(weightV.size(), 0);
        for (unsigned int i = 0, n = weightV.size(); i < n; ++i)
          weightSumMap_[wName][i] += (w0 * weightV[i]);
      }
      // incGenOnly(w0);
      //incPSOnly(w0, wPS);
    }

    void mergeSumMap(const Counter& other) {
      num_ += other.num_;
      sumw_ += other.sumw_;
      sumw2_ += other.sumw2_;
      //if weightMap_ for "this" is empty, create map elements with empty
      //vectors before merging
      if (weightSumMap_.empty() && !other.weightSumMap_.empty()) {
        for (auto& wmap : other.weightSumMap_) {
          std::vector<long double> temp;
          weightSumMap_.insert({wmap.first, temp});
        }
      }

      for (auto& wmap : weightSumMap_) {
        if (other.weightSumMap_.find(wmap.first) != other.weightSumMap_.end())
          mergeSumVectors(wmap.second, other.weightSumMap_.at(wmap.first));
      }
    }

    //private:
    // the counters
    long long num_ = 0;
    long double sumw_ = 0;
    long double sumw2_ = 0;
    std::map<std::string, std::vector<long double>> weightSumMap_;
  };

  struct CounterMap {
    std::map<std::string, Counter> countermap;
    Counter* active_el = nullptr;
    std::string active_label = "";

    void mergeSumMap(const CounterMap& other) {
      for (const auto& y : other.countermap) {
        countermap[y.first].mergeSumMap(y.second);
      }
      active_el = nullptr;
    }

    void clear() {
      for (auto x : countermap)
        x.second.clear();
    }

    void setLabel(std::string label) {
      active_el = &(countermap[label]);
      active_label = label;
    }
    void checkLabelSet() {
      if (!active_el)
        throw cms::Exception("LogicError", "Called CounterMap::get() before setting the active label\n");
    }
    Counter* get() {
      checkLabelSet();
      return active_el;
    }
    std::string& getLabel() {
      checkLabelSet();
      return active_label;
    }
  };

}  // namespace genCounter
#endif
