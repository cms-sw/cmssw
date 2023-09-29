#ifndef CondTools_RunInfo_LumiSectionFilter_h
#define CondTools_RunInfo_LumiSectionFilter_h

#include "CondFormats/Common/interface/TimeConversions.h"
#include "CondCore/CondDB/interface/Time.h"
#include <vector>
#include <memory>

template <class T>
struct LumiSectionFilter {
  LumiSectionFilter(const std::vector<std::pair<cond::Time_t, std::shared_ptr<T>>>& samples)
      : currLow(samples.begin()), currUp(samples.begin()), end(samples.end()) {
    currUp++;
  }

  void reset(const std::vector<std::pair<cond::Time_t, std::shared_ptr<T>>>& samples) {
    currLow = samples.begin();
    currUp = samples.begin();
    currUp++;
    end = samples.end();
    currentDipTime = 0;
  }

  bool process(cond::Time_t dipTime) {
    if (currLow == end)
      return false;
    bool search = false;
    if (currentDipTime == 0) {
      search = true;
    } else {
      if (dipTime == currentDipTime)
        return true;
      else {
        cond::Time_t upper = cond::time::MAX_VAL;
        if (currUp != end)
          upper = currUp->first;
        if (dipTime < upper && currentDipTime >= currLow->first)
          return false;
        else {
          search = true;
        }
      }
    }
    if (search) {
      while (currUp != end and currUp->first < dipTime) {
        currLow++;
        currUp++;
      }
      currentDipTime = dipTime;
      return currLow != end;
    }
    return false;
  }

  cond::Time_t currentSince() { return currLow->first; }
  T& currentPayload() { return *currLow->second; }

  typename std::vector<std::pair<cond::Time_t, std::shared_ptr<T>>>::const_iterator current() { return currLow; }
  typename std::vector<std::pair<cond::Time_t, std::shared_ptr<T>>>::const_iterator currLow;
  typename std::vector<std::pair<cond::Time_t, std::shared_ptr<T>>>::const_iterator currUp;
  typename std::vector<std::pair<cond::Time_t, std::shared_ptr<T>>>::const_iterator end;
  cond::Time_t currentDipTime = 0;
};

#endif