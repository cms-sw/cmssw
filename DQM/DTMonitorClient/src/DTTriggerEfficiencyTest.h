#ifndef DTTriggerEfficiencyTest_H
#define DTTriggerEfficiencyTest_H


/** \class DTTriggerEfficiencyTest
 * *
 *  DQM Test Client
 *
 *  \author  C. Battilana - CIEMAT
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
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
  void bookHistos(DQMStore::IBooker & ibooker,std::string hTag,std::string folder);

  /// Book the new MEs (for each wheel)
  void bookWheelHistos(DQMStore::IBooker & ibooker,int wheel,std::string hTag,std::string folder);

  /// Book the new MEs (for each chamber)
  void bookChambHistos(DQMStore::IBooker & ibooker,DTChamberId chambId, std::string htype , std::string folder = "");

  /// Get the ME name (by wheel)
  std::string getMEName(std::string histoTag, std::string folder, int wh);

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// DQM Client Diagnostic
  void runClientDiagnostic(DQMStore::IBooker &, DQMStore::IGetter &);
  void Bookings(DQMStore::IBooker &, DQMStore::IGetter &);

 private:

  std::map<std::string, MonitorElement*> globalEffDistr;
  std::map<int,std::map<std::string,MonitorElement*> > EffDistrPerWh;
  std::map<uint32_t,std::map<std::string,MonitorElement*> > chambME;
  DTTrigGeomUtils* trigGeomUtils;
  bool detailedPlots;

  bool bookingdone;

};

#endif
