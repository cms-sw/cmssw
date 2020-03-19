#ifndef TMarkov_H
#define TMarkov_H

#include "TObject.h"

class TMarkov : public TObject {
private:
  int fNPeakValues, fNbinu;
  int imax;
  double peak[3];
  double u[101], binu[102];

  void init();
  int computeChain(int *);

public:
  // Default Constructor, mainly for Root
  TMarkov();

  // Destructor: Does nothing
  ~TMarkov() override;

  void peakFinder(int *);
  double getPeakValue(int i) const { return peak[i]; }
  int getBinMax() const { return imax; }

  ClassDefOverride(TMarkov, 0)
};

#endif
