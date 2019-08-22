#ifndef ApvAnalysis_SimplePedestalCalculator_h
#define ApvAnalysis_SimplePedestalCalculator_h

#include "CalibTracker/SiStripAPVAnalysis/interface/TkPedestalCalculator.h"
#include <map>
/**
 * Concrete implementation of  TkPedestalCalculator for Simple.
 */

class SimplePedestalCalculator : public TkPedestalCalculator {
public:
  SimplePedestalCalculator(int evnt_ini);
  ~SimplePedestalCalculator() override;

  void resetPedestals() override {
    thePedestal.clear();
    theRawNoise.clear();
  }
  void setPedestals(ApvAnalysis::PedestalType& in) override { thePedestal = in; }
  void setRawNoise(ApvAnalysis::PedestalType& in) { theRawNoise = in; }

  void updateStatus() override;

  void updatePedestal(ApvAnalysis::RawSignalType& in) override;

  ApvAnalysis::PedestalType rawNoise() const override { return theRawNoise; }
  ApvAnalysis::PedestalType pedestal() const override { return thePedestal; }

  void newEvent() override;

private:
  void init();
  void initializePedestal(ApvAnalysis::RawSignalType& in);
  void refinePedestal(ApvAnalysis::RawSignalType& in);

protected:
  ApvAnalysis::PedestalType thePedestal;
  ApvAnalysis::PedestalType theRawNoise;
  std::vector<int> thePedSum, thePedSqSum;
  std::vector<unsigned short> theEventPerStrip;
  int numberOfEvents;
  int eventsRequiredToCalibrate;
  //  int eventsRequiredToUpdate;
  //  float cutToAvoidSignal;
  bool alreadyUsedEvent;
};
#endif
