#ifndef DQMClientExample_H
#define DQMClientExample_H

/** \class DQMClientExample
 * *
 *  DQM Test Client
 *
 *  $Date: 2007/10/11 22:41:13 $
 *  $Revision: 1.2 $
 *  \author  M. Zanetti CERN
 *   
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>


class DQMClientExample: public edm::EDAnalyzer {

public:

  /// Constructor
  DQMClientExample(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DQMClientExample();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// BeginRun
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// Fake Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) ;

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                            const edm::EventSetup& context) ;

  /// DQM Client Diagnostic
//  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
//                          const edm::EventSetup& c);

  /// EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c);

  /// Endjob
  void endJob();

private:

  void initialize();
  
  edm::ParameterSet parameters_;

  DaqMonitorBEInterface* dbe_;  
  std::string monitorName_;
  int counterLS_;      ///counter
  int counterEvt_;     ///counter
  int prescaleLS_;     ///units of lumi sections
  int prescaleEvt_;    ///prescale on number of events

  // -------- member data --------
  MonitorElement * clientHisto;

};

#endif


