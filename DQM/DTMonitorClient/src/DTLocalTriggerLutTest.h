#ifndef DTLocalTriggerLutTest_H
#define DTLocalTriggerLutTest_H


/** \class DTLocalTriggerLutTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2010/01/05 10:15:46 $
 *  $Revision: 1.6 $
 *  \author  C. Battilana S. Marcellini - INFN Bologna
 *   
 */


#include "DQM/DTMonitorClient/src/DTLocalTriggerBaseTest.h"



class DTLocalTriggerLutTest: public DTLocalTriggerBaseTest{

public:

  /// Constructor
  DTLocalTriggerLutTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTLocalTriggerLutTest();

  
protected:

  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// Run client analysis
  void runClientDiagnostic();

 private:

  /// Perform Lut Test logical operations
  int performLutTest(double mean,double RMS,double thresholdMean,double thresholdRMS);

  /// Fill summary plots managing double MB4 chambers
  void fillWhPlot(MonitorElement *plot,int sect,int stat, float value, bool lessIsBest = true);

  double thresholdPhiMean, thresholdPhibMean;
  double thresholdPhiRMS, thresholdPhibRMS;
  bool doCorrStudy;

};

#endif
