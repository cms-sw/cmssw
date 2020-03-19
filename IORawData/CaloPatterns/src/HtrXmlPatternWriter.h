#ifndef HtrXmlPatternWriter_h_included
#define HtrXmlPatternWriter_h_included 1

#include "HtrXmlPatternSet.h"
#include <ostream>

class HtrXmlPatternWriter {
public:
  HtrXmlPatternWriter();
  void setTagName(std::string tn) { m_tagName = tn; }
  void writePattern(HalfHtrData* spigotData, int fiber, std::ostream& os, int level = 0);

private:
  void packWordsStd(int adc0, int adc1, int adc2, int capid, unsigned int& w1, unsigned int& w2);
  std::string m_tagName, m_stamp;
};

#endif
