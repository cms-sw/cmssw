#ifndef ApvAnalysis_DBPedestal_h
#define ApvAnalysis_DBPedestal_h

#include "CalibTracker/SiStripAPVAnalysis/interface/TkPedestalCalculator.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/PedFactoryService.h"
#include<map>
/**
 * Concrete implementation of  DBPedestal.
 * this will retrieve ped from online DB and will pass to ApvAnalysis 
 */

class DBPedestal: public TkPedestalCalculator {
public: 

  DBPedestal(int detId, int thisApv);
  virtual ~DBPedestal();

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

  //  ApvAnalysisFactory*  apvFactory_;
 
  PedFactoryService* PedService_ ;
  const SiStripPedestals*  pedDB_;

  //  std::vector<double> thePedSum,thePedSqSum;  
  //  std::vector<unsigned short> theEventPerStrip;
  //  int eventsRequiredToCalibrate;
  //  int eventsRequiredToUpdate;
  // float cutToAvoidSignal;
  bool alreadyRetrievedFromDB;
  int numberOfEvents;
  int detId_;
  int thisApv_;


  };
#endif
