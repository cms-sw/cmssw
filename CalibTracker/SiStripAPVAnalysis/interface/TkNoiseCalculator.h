#ifndef TkNoiseCalculator_H
#define TkNoiseCalculator_H

#include "CalibTracker/SiStripAPVAnalysis/interface/ApvAnalysis.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TkStateMachine.h"

/**
 * The abstract class for noise calculation/subtraction.
 */
class TkNoiseCalculator {
 public:
  virtual ~TkNoiseCalculator() {}
  /** Return status flag indicating if noise values are usable */
  TkStateMachine* status() {return &theStatus;}
  
  virtual void setStripNoise(ApvAnalysis::PedestalType& in) = 0;
  /** Return reconstructed noise */
  virtual ApvAnalysis::PedestalType noise() const = 0;
  virtual float stripNoise(int) const = 0;
  
  /** Request that status flag be updated */
  virtual void updateStatus() = 0 ;
  
  virtual void resetNoise() = 0;
  
  //Actions

  /** Update noise with current event */
  virtual void updateNoise(ApvAnalysis::PedestalType&) = 0;
  /** Tell noise calculator that a new event is available */  
  virtual void newEvent(){}

 protected:
  TkStateMachine theStatus;  
};

#endif
