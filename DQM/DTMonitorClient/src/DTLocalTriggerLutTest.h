#ifndef DTLocalTriggerLutTest_H
#define DTLocalTriggerLutTest_H

/** \class DTLocalTriggerLutTest
 * *
 *  DQM Test Client
 *
 *  \author  C. Battilana S. Marcellini - INFN Bologna
 *   
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 */

#include "DQM/DTMonitorClient/src/DTLocalTriggerBaseTest.h"

class DTLocalTriggerLutTest : public DTLocalTriggerBaseTest {
public:
  /// Constructor
  DTLocalTriggerLutTest(const edm::ParameterSet& ps);

  /// Destructor
  ~DTLocalTriggerLutTest() override;

protected:
  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c) override;

  /// Run client analysis

  void runClientDiagnostic(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) override;
  void Bookings(DQMStore::IBooker&, DQMStore::IGetter&);

  const int wheelArrayShift = 3;

private:
  /// Perform Lut Test logical operations
  int performLutTest(double mean, double RMS, double thresholdMean, double thresholdRMS);

  /// Fill summary plots managing double MB4 chambers
  void fillWhPlot(MonitorElement* plot, int sect, int stat, float value, bool lessIsBest = true);

  double thresholdPhiMean, thresholdPhibMean;
  double thresholdPhiRMS, thresholdPhibRMS;
  bool doCorrStudy;

  bool bookingdone;
};

#endif
