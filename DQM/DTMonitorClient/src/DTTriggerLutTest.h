#ifndef DTTriggerLutTest_H
#define DTTriggerLutTest_H


/** \class DTTriggerLutTest
 * *
 *  DQM Test Client to monitor Local Trigger
 *  position / direction assignement
 *
 *  $Date: 2011/06/10 13:50:12 $
 *  $Revision: 1.1 $
 *  \author  D.Fasanella - INFN Bologna
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

  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// Run client analysis
  void runClientDiagnostic();

 private:

  /// Perform Lut Test logical operations
  int performLutTest(double perc,double threshold1,double threshold2);

  /// Fill summary plots managing double MB4 chambers
  void fillWhPlot(MonitorElement *plot,int sect,int stat, float value, bool lessIsBest = true);

  void bookCmsHistos1d(std::string hTag, std::string folder="");

  double thresholdWarnPhi, thresholdErrPhi;
  double thresholdWarnPhiB, thresholdErrPhiB;
  double validRange;
  bool   detailedAnalysis;	
	
};

#endif
