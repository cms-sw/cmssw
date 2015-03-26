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



class DTLocalTriggerTPTest: public DTLocalTriggerBaseTest{

public:

  /// Constructor
  DTLocalTriggerTPTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTLocalTriggerTPTest();

protected:

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// Run client analysis

  void runClientDiagnostic(DQMStore::IBooker &, DQMStore::IGetter &);

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &);

  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const &);

 private:

  bool bookingdone;

};

#endif
