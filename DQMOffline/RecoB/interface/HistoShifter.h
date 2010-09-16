#ifndef HistoShifter_H
#define HistoShifter_H

#include "TH1F.h"

class TH1F;

class HistoShifter {
 public:
  HistoShifter (){}
  ~HistoShifter(){}
  bool insertAndShift(TH1F * in, float value);
  bool insertAndShift(TH1F * in, float value, float error);
};

#endif
