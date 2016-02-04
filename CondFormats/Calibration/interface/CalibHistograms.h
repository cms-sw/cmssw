#ifndef CalibHistograms_h
#define CalibHistograms_h
#include<vector>
struct CalibHistogram{
  CalibHistogram(){}
  std::vector<float> m_histo;
};
struct CalibHistograms{
  CalibHistograms(){}
  std::vector<CalibHistogram> m_data;
};
#endif
