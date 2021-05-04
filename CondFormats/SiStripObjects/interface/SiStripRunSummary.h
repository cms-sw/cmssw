#ifndef SiStripRunSummary_h
#define SiStripRunSummary_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <string>
#include <iostream>

class SiStripRunSummary {
public:
  SiStripRunSummary(){};
  ~SiStripRunSummary(){};

  bool put(std::string runSummary) {
    runSummary_ = runSummary;
    return true;
  }
  std::string getRunSummary() const { return runSummary_; }

private:
  std::string runSummary_;

  COND_SERIALIZABLE;
};

#endif
