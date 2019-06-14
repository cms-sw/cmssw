#ifndef CalibHistograms_h
#define CalibHistograms_h
#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
struct CalibHistogram {
  CalibHistogram() {}
  std::vector<float> m_histo;

  COND_SERIALIZABLE;
};
struct CalibHistograms {
  CalibHistograms() {}
  std::vector<CalibHistogram> m_data;

  COND_SERIALIZABLE;
};
#endif
