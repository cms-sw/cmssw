#ifndef ApvAnalysis_SimpleNoiseCalculator_H
#define ApvAnalysis_SimpleNoiseCalculator_H

#include "CalibTracker/SiStripAPVAnalysis/interface/TkNoiseCalculator.h"
/**
 * Concrete implementation of TkNoiseCalculator  for Simple.
 */

class SimpleNoiseCalculator : public TkNoiseCalculator {
  
public:  
  
  // Use the constructor without arguments, since the other will soon
  // be obsolete.
  SimpleNoiseCalculator();
  SimpleNoiseCalculator(int evnt_ini, bool useDB);
  ~SimpleNoiseCalculator() override;

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
  bool useDB_;  

  int eventsRequiredToCalibrate_;
  // int eventsRequiredToUpdate_;
  // float cutToAvoidSignal_;
};
#endif
