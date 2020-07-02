#include "CalibTracker/SiStripAPVAnalysis/interface/TkCommonMode.h"
using namespace std;
vector<float> TkCommonMode::toVector() const {
  vector<float> temp;
  temp.reserve(myTkCommonModeTopology->numberOfStrips());
  for (int i = 0; i < myTkCommonModeTopology->numberOfStrips(); i++) {
    temp.push_back(returnAsVector()[myTkCommonModeTopology->setOfStrip(i)]);
  }
  return temp;
}
