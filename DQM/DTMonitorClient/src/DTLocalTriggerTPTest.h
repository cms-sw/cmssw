#ifndef DTLocalTriggerTPTest_H
#define DTLocalTriggerTPTest_H


/** \class DTLocalTriggerTPTest
 * *
 *  DQM Test Client
 *
 *  \author  C. Battilana S. Marcellini - INFN Bologna
 *   
 */


#include "DQM/DTMonitorClient/src/DTLocalTriggerBaseTest.h"



class DTLocalTriggerTPTest: public DTLocalTriggerBaseTest{

public:

  /// Constructor
  DTLocalTriggerTPTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTLocalTriggerTPTest();

protected:

  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// Run client analysis
  void runClientDiagnostic();

 private:

};

#endif
