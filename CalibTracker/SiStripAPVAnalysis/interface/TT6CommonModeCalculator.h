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
    
  TT6CommonModeCalculator(TkNoiseCalculator* noise_calc,
                          TkApvMask* mask_calc, float sig_cut);

  virtual ~TT6CommonModeCalculator();

  ApvAnalysis::PedestalType doIt(const ApvAnalysis::PedestalType&); 
  
  void setCM(TkCommonMode* in) {theTkCommonMode = in;}
  void setCM(const std::vector<float>& in) {theTkCommonMode->setCommonMode(in);}
  TkCommonMode* commonMode() {return theTkCommonMode;}

  void newEvent();
  float getCMSlope() { return slope;}
  
protected:
  
  void calculateCommonMode(ApvAnalysis::PedestalType&);
  void calculateCMSlope(ApvAnalysis::PedestalType&);
  
  TkCommonMode*        theTkCommonMode;
  std::vector<float>        theCommonModeValues;
  TkNoiseCalculator*   theNoiseCalculator;
  TkApvMask*           theApvMask;
  bool alreadyUsedEvent;
  float slope;


  float cutToAvoidSignal;
}; 
#endif




