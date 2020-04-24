#ifndef DTLocalTriggerEfficiencyTest_H
#define DTLocalTriggerEfficiencyTest_H


/** \class DTLocalTriggerEfficiencyTest
 * *
 *  DQM Test Client
 *
 *  \author  C. Battilana S. Marcellini - INFN Bologna
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah ncpp-um-my
 *
 *   
 */


#include "DQM/DTMonitorClient/src/DTLocalTriggerBaseTest.h"


class DTTrigGeomUtils;

class DTLocalTriggerEfficiencyTest: public DTLocalTriggerBaseTest{

public:

  /// Constructor
  DTLocalTriggerEfficiencyTest(const edm::ParameterSet& ps);
  
  /// Destructor
  ~DTLocalTriggerEfficiencyTest() override;

protected:

  /// Book the new MEs (for each chamber)
  void bookChambHistos(DQMStore::IBooker &, DTChamberId chambId, std::string htype );

  /// Compute efficiency plots
  void makeEfficiencyME(TH1D* numerator, TH1D* denominator, MonitorElement* result);

  /// Compute 2D efficiency plots
  void makeEfficiencyME2D(TH2F* numerator, TH2F* denominator, MonitorElement* result);

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c) override;


  /// DQM Client Diagnostic

  void runClientDiagnostic(DQMStore::IBooker &, DQMStore::IGetter &) override;
  void Bookings(DQMStore::IBooker &, DQMStore::IGetter &);

  const int wheelArrayShift = 3;

 private:

  std::map<uint32_t,std::map<std::string,MonitorElement*> > chambME;
  DTTrigGeomUtils *trigGeomUtils;

  bool bookingdone;

};

#endif
