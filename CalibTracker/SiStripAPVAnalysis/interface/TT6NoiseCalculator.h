#ifndef ApvAnalysis_TT6NoiseCalculator_H
#define ApvAnalysis_TT6NoiseCalculator_H

#include "CalibTracker/SiStripAPVAnalysis/interface/TkNoiseCalculator.h"
/**
 * Concrete implementation of TkNoiseCalculator  for TT6.
 */

class TT6NoiseCalculator : public TkNoiseCalculator {
  
public:  
  
  // Use the constructor without arguments, since the other will soon
  // be obsolete.
  TT6NoiseCalculator();
  TT6NoiseCalculator(int evnt_ini,int evnt_iter,float sig_cut);
  ~TT6NoiseCalculator() override;

  void setStripNoise(ApvAnalysis::PedestalType& in) override {theNoise.clear(); theNoise = in;}
  ApvAnalysis::PedestalType noise() const override {return theNoise;}
  float stripNoise(int in) const override {return theNoise[in];}
  int nevents() const {return numberOfEvents;}

  void updateStatus() override;  
  void resetNoise() override {theNoise.clear();}
  void updateNoise(ApvAnalysis::PedestalType& in) override;
  void newEvent() override;
  
  ApvAnalysis::PedestalType stripCMPSubtractedSignal() const
               {return theCMPSubtractedSignal;}
  
protected:
  void init();

protected:
  ApvAnalysis::PedestalType theNoise;
  ApvAnalysis::PedestalType theCMPSubtractedSignal;
  std::vector<double> theNoiseSum,theNoiseSqSum;
  std::vector<unsigned short> theEventPerStrip;
  int numberOfEvents;
  bool alreadyUsedEvent;

  int eventsRequiredToCalibrate_;
  int eventsRequiredToUpdate_;
  float cutToAvoidSignal_;
};
#endif
