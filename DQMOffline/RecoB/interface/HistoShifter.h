#ifndef HistoShifter_H
#define HistoShifter_H

#include "TH1F.h"

class HistoShifter {
 public:
  HistoShifter (){}
  ~HistoShifter(){}
  bool insertAndShift(TH1F * in, const float& value);
  bool insertAndShift(TH1F * in, const float& value, const float& error);
};

#endif
