#ifndef DQMSourceExample_H
#define DQMSourceExample_H

/** \class DQMSourceExample
 * *
 *  DQM Test Client
 *
 *  $Date: 2007/10/11 22:41:13 $
 *  $Revision: 1.4 $
 *  \author  M. Zanetti CERN
 *   
 */
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//
// class declaration
//

class DQMSourceExample : public edm::EDAnalyzer {
public:
  DQMSourceExample( const edm::ParameterSet& );
  ~DQMSourceExample();

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
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);

  /// EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c);

  /// Endjob
  void endJob();

private:
 
  edm::ParameterSet parameters_;

  DaqMonitorBEInterface* dbe_;  
  std::string monitorName_;
  int counterEvt_;      ///counter
  int prescaleEvt_;     ///every n events
                        /// FIXME, make prescale module?

  // ----------member data ---------------------------

  MonitorElement * h1;
  MonitorElement * h2;
  MonitorElement * h3;
  MonitorElement * h4;
  MonitorElement * h5;
  MonitorElement * h6;
  MonitorElement * h7;
  MonitorElement * h8;
  MonitorElement * h9;
  MonitorElement * i1;
  MonitorElement * f1;
  MonitorElement * s1;
  MonitorElement * p1;
  float XMIN; float XMAX;
};

#endif

