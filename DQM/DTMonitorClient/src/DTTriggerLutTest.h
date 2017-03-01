#ifndef DTTriggerLutTest_H
#define DTTriggerLutTest_H


/** \class DTTriggerLutTest
 * *
 *  DQM Test Client to monitor Local Trigger
 *  position / direction assignement
 *
 *  \author  D.Fasanella - INFN Bologna
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 *   
 */


#include "DQM/DTMonitorClient/src/DTLocalTriggerBaseTest.h"

class TSpectrum;

class DTTriggerLutTest: public DTLocalTriggerBaseTest{

public:

  /// Constructor
  DTTriggerLutTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTTriggerLutTest();

protected:

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);
  void Bookings(DQMStore::IBooker &, DQMStore::IGetter &);


  /// Run client analysis
  void runClientDiagnostic(DQMStore::IBooker &, DQMStore::IGetter &);

 private:

  /// Perform Lut Test logical operations
  int performLutTest(double perc,double threshold1,double threshold2);

  /// Fill summary plots managing double MB4 chambers
  void fillWhPlot(MonitorElement *plot,int sect,int stat, float value, bool lessIsBest = true);

  void bookCmsHistos1d(DQMStore::IBooker &,std::string hTag, std::string folder="");

  double thresholdWarnPhi, thresholdErrPhi;
  double thresholdWarnPhiB, thresholdErrPhiB;
  double validRange;
  bool   detailedAnalysis;	
  
  bool bookingdone;
	
};

#endif
