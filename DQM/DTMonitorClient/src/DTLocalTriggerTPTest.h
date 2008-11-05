#ifndef DTLocalTriggerTPTest_H
#define DTLocalTriggerTPTest_H


/** \class DTLocalTriggerTPTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/10/07 14:26:43 $
 *  $Revision: 1.13 $
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

  /// Begin Job
  void beginJob(const edm::EventSetup& c);

  /// Run client analysis
  void runClientDiagnostic();

 private:

};

#endif
