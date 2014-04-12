#ifndef Tracker_TkCommonModeCalculator_H
#define Tracker_TkCommonModeCalculator_H

#include "CalibTracker/SiStripAPVAnalysis/interface/ApvAnalysis.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TkCommonMode.h"
/**
 * The abstract class for common mode subtraction.
 */
class TkCommonModeCalculator{
  
 public:  
  virtual ~TkCommonModeCalculator() {}
  /** Return CM-subtracted data in APV */
  virtual ApvAnalysis::PedestalType doIt(const ApvAnalysis::PedestalType&) = 0 ;  
  virtual void setCM(TkCommonMode*) = 0;
  virtual void setCM(const std::vector<float>&) = 0;
  /** Get CM value */
  virtual TkCommonMode* commonMode() = 0;
  /** Tell CM calculator that a new event is available */
  virtual void newEvent() {}
  /** Get Slope */
  virtual float getCMSlope() = 0;
};

#endif
