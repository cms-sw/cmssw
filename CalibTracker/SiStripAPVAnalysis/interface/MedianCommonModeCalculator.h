#ifndef Tracker_MedianCommonModeCalculator_h
#define Tracker_MedianCommonModeCalculator_h

#include "CalibTracker/SiStripAPVAnalysis/interface/TkCommonModeCalculator.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TkNoiseCalculator.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TkApvMask.h"
/**
 * Concrete implementation of TkCommonModeCalculator  for Median.
 */

class MedianCommonModeCalculator : public TkCommonModeCalculator {
public:
  MedianCommonModeCalculator();

  ~MedianCommonModeCalculator() override;

  ApvAnalysis::PedestalType doIt(const ApvAnalysis::PedestalType&) override;

  void setCM(TkCommonMode* in) override { theTkCommonMode = in; }
  void setCM(const std::vector<float>& in) override { theTkCommonMode->setCommonMode(in); }
  TkCommonMode* commonMode() override { return theTkCommonMode; }

  void newEvent() override;
  float getCMSlope() override { return slope; }

protected:
  void calculateCommonMode(ApvAnalysis::PedestalType&);

  TkCommonMode* theTkCommonMode;
  std::vector<float> theCommonModeValues;
  bool alreadyUsedEvent;
  float slope;

  ///  TkNoiseCalculator*   theNoiseCalculator;
  ///  TkApvMask*           theApvMask;
  ///  float cutToAvoidSignal;
};
#endif
