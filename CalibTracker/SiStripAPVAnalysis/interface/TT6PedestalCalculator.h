#ifndef ApvAnalysis_TT6PedestalCalculator_h
#define ApvAnalysis_TT6PedestalCalculator_h

#include "CalibTracker/SiStripAPVAnalysis/interface/TkPedestalCalculator.h"
#include<map>
/**
 * Concrete implementation of  TkPedestalCalculator for TT6.
 */

class TT6PedestalCalculator: public TkPedestalCalculator {
public: 

  TT6PedestalCalculator(int evnt_ini, int evnt_iter, float sig_cut);
  virtual ~TT6PedestalCalculator();



  void resetPedestals() {
    thePedestal.clear();
    theRawNoise.clear();
  } 
  void setPedestals (ApvAnalysis::PedestalType& in) {thePedestal=in;}
  void setRawNoise (ApvAnalysis::PedestalType& in) {theRawNoise=in;}
    
  void updateStatus();

  void updatePedestal (ApvAnalysis::RawSignalType& in);

  ApvAnalysis::PedestalType rawNoise() const { return theRawNoise;}
  ApvAnalysis::PedestalType  pedestal() const { return thePedestal;}

 
  void newEvent();

private:
  void init();
  void initializePedestal (ApvAnalysis::RawSignalType& in);
  void refinePedestal (ApvAnalysis::RawSignalType& in);

protected:
  ApvAnalysis::PedestalType thePedestal;
  ApvAnalysis::PedestalType theRawNoise;
  std::vector<double> thePedSum,thePedSqSum;  
  std::vector<unsigned short> theEventPerStrip;
  int numberOfEvents;
  int eventsRequiredToCalibrate;
  int eventsRequiredToUpdate;
  float cutToAvoidSignal;
  bool alreadyUsedEvent;

  };
#endif
