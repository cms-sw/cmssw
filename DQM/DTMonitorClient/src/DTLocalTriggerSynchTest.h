#ifndef DTLocalTriggerSynchTest_H
#define DTLocalTriggerSynchTest_H


/** \class DTLocalTriggerSynchTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2009/10/16 08:40:03 $
 *  $Revision: 1.1 $
 *  \author  C. Battilana - CIEMAT
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
  void bookChambHistos(DTChamberId chambId, std::string htype, std::string subfolder="");

  /// Compute efficiency plots
  void makeRatioME(TH1F* numerator, TH1F* denominator, MonitorElement* result);

  /// Get float MEs
  float getFloatFromME(DTChamberId chId, std::string meType);

  /// Begin Job
  void beginJob(const edm::EventSetup& c);

  /// begin Run
  void beginRun(const edm::Run& run, const edm::EventSetup& c);

  /// End Job
  void endJob();

  /// DQM Client Diagnostic
  void runClientDiagnostic();

 private:

  std::map<uint32_t,std::map<std::string,MonitorElement*> > chambME;
  std::string numHistoTag;
  std::string denHistoTag;
  std::string ratioHistoTag;
  double bxTime;
  bool rangeInBX;
  bool writeDB;
  DTTPGParameters wPhaseMap;

};

#endif
