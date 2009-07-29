#ifndef DTLocalTriggerLutTest_H
#define DTLocalTriggerLutTest_H


/** \class DTLocalTriggerLutTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/10/07 14:26:43 $
 *  $Revision: 1.3 $
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

  /// Begin Job
  void beginJob(const edm::EventSetup& c);

  /// Run client analysis
  void runClientDiagnostic();

 private:

  /// Perform Lut Test logical operations
  int performLutTest(double mean,double RMS,double thresholdMean,double thresholdRMS);

  /// Fill summary plots managing double MB4 chambers
  void fillWhPlot(MonitorElement *plot,int sect,int stat, float value, bool lessIsBest = true);

  double thresholdPhiMean, thresholdPhibMean;
  double thresholdPhiRMS, thresholdPhibRMS;

};

#endif
