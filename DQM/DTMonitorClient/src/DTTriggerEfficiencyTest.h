#ifndef DTTriggerEfficiencyTest_H
#define DTTriggerEfficiencyTest_H


/** \class DTTriggerEfficiencyTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2011/06/15 14:18:43 $
 *  $Revision: 1.3 $
 *  \author  C. Battilana - CIEMAT
 *   
 */


#include "DQM/DTMonitorClient/src/DTLocalTriggerBaseTest.h"

#include <string>

class DTTrigGeomUtils;

class DTTriggerEfficiencyTest: public DTLocalTriggerBaseTest{

public:

  /// Constructor
  DTTriggerEfficiencyTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTTriggerEfficiencyTest();

protected:

  /// Compute 1D/2D efficiency plots
  void makeEfficiencyME(TH2F* numerator, TH2F* denominator, MonitorElement* result2DWh, MonitorElement* result1DWh, MonitorElement* result1D);

  /// Compute 2D efficiency plots
  void makeEfficiencyME(TH2F* numerator, TH2F* denominator, MonitorElement* result2DWh);

  /// Book the new MEs (global)
  void bookHistos(std::string hTag,std::string folder);

  /// Book the new MEs (for each wheel)
  void bookWheelHistos(int wheel,std::string hTag,std::string folder);

  /// Book the new MEs (for each chamber)
  void bookChambHistos(DTChamberId chambId, std::string htype , std::string folder = "");

  /// Get the ME name (by wheel)
  std::string getMEName(std::string histoTag, std::string folder, int wh);

  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// DQM Client Diagnostic
  void runClientDiagnostic();



 private:

  std::map<std::string, MonitorElement*> globalEffDistr;
  std::map<int,std::map<std::string,MonitorElement*> > EffDistrPerWh;
  std::map<uint32_t,std::map<std::string,MonitorElement*> > chambME;
  DTTrigGeomUtils* trigGeomUtils;
  bool detailedPlots;

};

#endif
