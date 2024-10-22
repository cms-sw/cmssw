#ifndef Tracker_TT6CommonModeCalculator_h
#define Tracker_TT6CommonModeCalculator_h

#include "CalibTracker/SiStripAPVAnalysis/interface/TkCommonModeCalculator.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TkNoiseCalculator.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TkApvMask.h"
/**
 * Concrete implementation of TkCommonModeCalculator  for TT6.
 */

class TT6CommonModeCalculator : public TkCommonModeCalculator {
public:
  TT6CommonModeCalculator(TkNoiseCalculator* noise_calc, TkApvMask* mask_calc, float sig_cut);

  ~TT6CommonModeCalculator() override;

  ApvAnalysis::PedestalType doIt(const ApvAnalysis::PedestalType&) override;

  void setCM(TkCommonMode* in) override { theTkCommonMode = in; }
  void setCM(const std::vector<float>& in) override { theTkCommonMode->setCommonMode(in); }
  TkCommonMode* commonMode() override { return theTkCommonMode; }

  void newEvent() override;
  float getCMSlope() override { return slope; }

protected:
  void calculateCommonMode(ApvAnalysis::PedestalType&);
  void calculateCMSlope(ApvAnalysis::PedestalType&);

  TkCommonMode* theTkCommonMode;
  std::vector<float> theCommonModeValues;
  TkNoiseCalculator* theNoiseCalculator;
  TkApvMask* theApvMask;
  bool alreadyUsedEvent;
  float slope;

  float cutToAvoidSignal;
};
#endif
