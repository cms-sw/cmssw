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
  virtual ~SimpleNoiseCalculator();

  void setStripNoise(ApvAnalysis::PedestalType& in) {theNoise.clear(); theNoise = in;}
  ApvAnalysis::PedestalType noise() const {return theNoise;}
  float stripNoise(int in) const {return theNoise[in];}
  int nevents() const {return numberOfEvents;}

  void updateStatus();  
  void resetNoise() {theNoise.clear();}
  void updateNoise(ApvAnalysis::PedestalType& in);
  void newEvent();
  
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
