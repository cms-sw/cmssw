#ifndef DTLocalTriggerSynchTest_H
#define DTLocalTriggerSynchTest_H


/** \class DTLocalTriggerSynchTest
 * *
 *  DQM Test Client
 *
 *  $Date: 2009/08/03 16:10:23 $
 *  $Revision: 1.4 $
 *  \author  C. Battilana - CIEMAT
 *   
 */


#include "DQM/DTMonitorClient/src/DTLocalTriggerBaseTest.h"


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

 /*  /// Find histo Maximum using a fit */
/*   float findMaximum(TH1F* histo); */

  /// Begin Job
  void beginJob(const edm::EventSetup& c);

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

};

#endif
