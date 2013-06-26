#ifndef DTLocalTriggerEfficiencyTest_H
#define DTLocalTriggerEfficiencyTest_H


/** \class DTLocalTriggerEfficiencyTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2010/01/05 10:15:46 $
 *  $Revision: 1.5 $
 *  \author  C. Battilana S. Marcellini - INFN Bologna
 *   
 */


#include "DQM/DTMonitorClient/src/DTLocalTriggerBaseTest.h"


class DTTrigGeomUtils;

class DTLocalTriggerEfficiencyTest: public DTLocalTriggerBaseTest{

public:

  /// Constructor
  DTLocalTriggerEfficiencyTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTLocalTriggerEfficiencyTest();

protected:

  /// Book the new MEs (for each chamber)
  void bookChambHistos(DTChamberId chambId, std::string htype );

  /// Compute efficiency plots
  void makeEfficiencyME(TH1D* numerator, TH1D* denominator, MonitorElement* result);

  /// Compute 2D efficiency plots
  void makeEfficiencyME2D(TH2F* numerator, TH2F* denominator, MonitorElement* result);

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// DQM Client Diagnostic
  void runClientDiagnostic();



 private:

  std::map<uint32_t,std::map<std::string,MonitorElement*> > chambME;
  DTTrigGeomUtils *trigGeomUtils;

};

#endif
