#ifndef DTLocalTriggerTest_H
#define DTLocalTriggerTest_H


/** \class DTLocalTriggerTest
 * *
 *  DQM Test Client
 *
 *  \author  C. Battilana S. Marcellini - INFN Bologna
 *   
 */


#include "DQM/DTMonitorClient/src/DTLocalTriggerBaseTest.h"



class DTLocalTriggerTest: public DTLocalTriggerBaseTest{

public:

  /// Constructor
  DTLocalTriggerTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTLocalTriggerTest();

protected:

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  void Bookings(DQMStore::IBooker &, DQMStore::IGetter &);

  /// Run client analysis

  void runClientDiagnostic(DQMStore::IBooker &,DQMStore::IGetter &);

  void fillGlobalSummary(DQMStore::IGetter &);

 private:

  int nMinEvts;

  bool bookingdone;  
  

};

#endif
