#ifndef ApvAnalysis_TT6APVMask_H
#define ApvAnalysis_TT6APVMask_H

#include "CalibTracker/SiStripAPVAnalysis/interface/TkApvMask.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TkNoiseCalculator.h"
/**
 * Concrete implementation of TkApvMask  for TT6.
 */

class TT6ApvMask : public TkApvMask {  
public:

  // Use the first constructor, as the second one will soon
  // be obsolete.
  TT6ApvMask( int ctype, float ncut, float dcut, float tcut);
  ~TT6ApvMask() override;  

  void setMask(const MaskType& in) override {theMask_ = in;}
  MaskType mask() override {return theMask_;}
  
  void calculateMask(const ApvAnalysis::PedestalType&) override;

protected:
  bool defineNoisy(float avrg, float rms,float noise);

private:
  MaskType theMask_;
  int theCalculationFlag_;
  float theNoiseCut_;
  float theDeadCut_;
  float theTruncationCut_;
};

#endif











