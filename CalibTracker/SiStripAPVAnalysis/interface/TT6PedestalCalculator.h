#ifndef ApvAnalysis_TT6PedestalCalculator_h
#define ApvAnalysis_TT6PedestalCalculator_h

#include "CalibTracker/SiStripAPVAnalysis/interface/TkPedestalCalculator.h"
#include <map>
/**
 * Concrete implementation of  TkPedestalCalculator for TT6.
 */

class TT6PedestalCalculator : public TkPedestalCalculator {
public:
  TT6PedestalCalculator(int evnt_ini, int evnt_iter, float sig_cut);
  ~TT6PedestalCalculator() override;

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
  std::vector<double> thePedSum, thePedSqSum;
  std::vector<unsigned short> theEventPerStrip;
  int numberOfEvents;
  int eventsRequiredToCalibrate;
  int eventsRequiredToUpdate;
  float cutToAvoidSignal;
  bool alreadyUsedEvent;
};
#endif
