#ifndef DTLocalTriggerTPTest_H
#define DTLocalTriggerTPTest_H

/** \class DTLocalTriggerTPTest
 * *
 *  DQ Test Client
 *
 *  \author  C. Battilana S. Marcellini - INFN Bologna
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 *   
 */

#include "DQM/DTMonitorClient/src/DTLocalTriggerBaseTest.h"

class DTLocalTriggerTPTest : public DTLocalTriggerBaseTest {
public:
  /// Constructor
  DTLocalTriggerTPTest(const edm::ParameterSet &ps);

  /// Destructor
  ~DTLocalTriggerTPTest() override;

protected:
  /// BeginRun
  void beginRun(const edm::Run &r, const edm::EventSetup &c) override;

  /// Run client analysis

  void runClientDiagnostic(DQMStore::IBooker &, DQMStore::IGetter &) override;
  void Bookings(DQMStore::IBooker &, DQMStore::IGetter &);

private:
  bool bookingdone;
};

#endif
