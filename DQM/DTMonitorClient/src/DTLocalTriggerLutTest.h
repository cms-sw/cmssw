#ifndef DTLocalTriggerLutTest_H
#define DTLocalTriggerLutTest_H


/** \class DTLocalTriggerLutTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/05/22 10:49:59 $
 *  $Revision: 1.1 $
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

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);

};

#endif
