#ifndef Tracker_TkPedestalCalculator_h
#define Tracker_TkPedestalCalculator_h

#include "CalibTracker/SiStripAPVAnalysis/interface/ApvAnalysis.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TkStateMachine.h"
/**
 * The abstract class for pedestal calculation/subtraction.
 */
class TkPedestalCalculator{
 public:
  
  virtual ~TkPedestalCalculator() {}
  /** Return reconstructed pedestals */
  //  virtual ApvAnalysis::PedestalType pedestal() const = 0 ;
  virtual ApvAnalysis::PedestalType pedestal() const =0;
  virtual ApvAnalysis::PedestalType rawNoise() const=0;
  
  /** Return status flag indicating if pedestals are usable */
  TkStateMachine* status() {return &theStatus;}
  
  virtual void resetPedestals() = 0;
  virtual void setPedestals (ApvAnalysis::PedestalType&) = 0;

  virtual void setNoise( ApvAnalysis::PedestalType &) {}
  
  /** Request that status flag be updated */
  virtual void updateStatus() = 0 ;
  
  //
  // Actions
  //
  
  /** Update pedestals with current event */
  virtual void updatePedestal (ApvAnalysis::RawSignalType& in) = 0 ;

  /** Return raw noise, determined without CMN subtraction */

  /** Tell pedestal calculator that a new event is available */
  virtual void newEvent(){}
  
 protected:
  
  TkStateMachine theStatus;
  
};

#endif
