#ifndef DQMClientPhiSym_H
#define DQMClientPhiSym_H

/** \class DQMClientPhiSym
 * *
 *  DQM Client for phi symmetry
 *
 *  $Date: 2008/04/28 $
 *  $Revision: 1.1 $
 *  \author Andrea Gozzelino - Università e INFN Torino
 *   
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <string>


class DQMClientPhiSym: public edm::EDAnalyzer {

public:

  /// Constructor
  DQMClientPhiSym(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DQMClientPhiSym();

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

  DQMStore* dbe_;  
  std::string monitorName_;
  int counterLS_;      ///counter
  int counterEvt_;     ///counter
  int prescaleLS_;     ///units of lumi sections
  int prescaleEvt_;    ///prescale on number of events

  // -------- member data --------
  MonitorElement * clientHisto;

};

#endif


