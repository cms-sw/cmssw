#ifndef DTLocalTriggerSynchTest_H
#define DTLocalTriggerSynchTest_H


/** \class DTLocalTriggerSynchTest
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
#include "CondFormats/DTObjects/interface/DTTPGParameters.h"

class DTTrigGeomUtils;

class DTLocalTriggerSynchTest: public DTLocalTriggerBaseTest{

public:

  /// Constructor
  DTLocalTriggerSynchTest(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTLocalTriggerSynchTest();

protected:

  /// Book the new MEs (for each chamber)

  void bookChambHistos(DQMStore::IBooker &,DTChamberId chambId, std::string htype, std::string subfolder="");

  /// Compute efficiency plots
  void makeRatioME(TH1F* numerator, TH1F* denominator, MonitorElement* result);

  /// Get float MEs

  float getFloatFromME(DQMStore::IGetter &,DTChamberId chId, std::string meType);

  /// begin Run
  void beginRun(const edm::Run& run, const edm::EventSetup& c);


  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &);

  /// DQM Client Diagnostic

  void runClientDiagnostic(DQMStore::IBooker &, DQMStore::IGetter &);

  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, 
                          edm::LuminosityBlock const &, edm::EventSetup const &);

 private:

  std::map<uint32_t,std::map<std::string,MonitorElement*> > chambME;
  std::string numHistoTag;
  std::string denHistoTag;
  std::string ratioHistoTag;
  double bxTime;
  bool rangeInBX;
  int nBXLow;
  int nBXHigh;
  int minEntries;
  bool writeDB;
  DTTPGParameters wPhaseMap;

  bool bookingdone;

};

#endif
