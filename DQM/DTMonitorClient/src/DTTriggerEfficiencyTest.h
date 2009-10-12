#ifndef DTTriggerEfficiencyTest_H
#define DTTriggerEfficiencyTest_H


/** \class DTTriggerEfficiencyTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2008/10/07 14:26:43 $
 *  $Revision: 1.3 $
 *  \author  C. Battilana - CIEMAT
 *   
 */


#include "DQM/DTMonitorClient/src/DTLocalTriggerBaseTest.h"

class DTTrigGeomUtils;

class DTTriggerEfficiencyTest: public DTLocalTriggerBaseTest{

public:

  /// Constructor
  DTTriggerEfficiencyTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTTriggerEfficiencyTest();

protected:

  /// Book the new MEs (for each chamber)
  void bookChambHistos(DTChamberId chambId, std::string htype , std::string folder = "");

  /// Compute 2D efficiency plots
  void makeEfficiencyME2D(TH2F* numerator, TH2F* denominator, MonitorElement* result);

  /// Begin Job
  void beginJob(const edm::EventSetup& c);

  /// DQM Client Diagnostic
  void runClientDiagnostic();



 private:

  std::map<uint32_t,std::map<std::string,MonitorElement*> > chambME;
  DTTrigGeomUtils* trigGeomUtils;
  bool detailedPlots;

};

#endif
