#include "CalibTracker/SiStripAPVAnalysis/interface/TkCommonMode.h"
using namespace std;
vector<float> TkCommonMode::toVector() const {
  vector<float> temp;
  for (int i=0; i<myTkCommonModeTopology->numberOfStrips(); i++){
    temp.push_back(returnAsVector()[myTkCommonModeTopology->setOfStrip(i)]);
  }
  return temp;
}
