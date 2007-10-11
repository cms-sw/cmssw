#ifndef DQMAnalyzer_H
#define DQMAnalyzer_H

/*
 * \file DQMAnalyzer.h
 *
 * $Date: 2007/09/23 15:22:49 $
 * $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/Framework/interface/Run.h"
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElementBaseT.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sys/time.h>

using namespace edm;

class DQMAnalyzer: public EDAnalyzer{

public:

  /// Constructors  
  DQMAnalyzer(const ParameterSet& ps);
  DQMAnalyzer();
  
  /// Destructor
  virtual ~DQMAnalyzer();

protected:

  /// to be used by derived class

  /// BeginJob
  void beginJob(const EventSetup& c);

  /// Endjob
  void endJob(void);
  
  /// BeginRun
  void beginRun(const Run& run, const EventSetup& c);

  /// EndRun
  void endRun(const Run& run, const EventSetup& c);

  
  /// Begin LumiBlock
  void beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
                            const EventSetup& c) ;

  /// End LumiBlock
  /// DQM Client Diagnostic should be performed here
  void endLuminosityBlock(const LuminosityBlock& lumiSeg, 
                          const EventSetup& c);

  // Reset
  void reset(void);

  /// Analyze
  void analyze(const Event& e, const EventSetup& c) ;

  /// Save DQM output file
  void save(std::string flag="");

  /// Boolean prescale test for this event
  bool prescale();

  // FIXME: put additional methods here
  // processME();

  // private member accessors
  int getNumEvents(){return nevt_;}
  int getNumLumiSecs(){return nlumisecs_;}


  // environment variables
  int irun_,ilumisec_,ievent_,itime_;
  bool actonLS_ ;
  std::string rootFolder_;

  DaqMonitorBEInterface* dbe_;  
  ParameterSet parameters_;


  /********************************************************/
  //  The following member variables can be specified in  //
  //  the configuration input file for the process.       //
  /********************************************************/

  /// Prescale variables for restricting the frequency of analyzer
  /// behavior.  The base class does not implement prescales.
  /// Set to -1 to be ignored.
  int prescaleEvt_;    ///units of events
  int prescaleLS_;     ///units of lumi sections
  int prescaleTime_;   ///units of minutes
  int prescaleUpdate_; ///units of "updates", TBD

  /// The name of the monitoring process which derives from this
  /// class, used to standardize filename and file structure
  std::string monitorName_;

  /// Verbosity switch used for debugging or informational output
  bool debug_ ;

private:
  void initialize();

  /// framework ME
  MonitorElement * runId_;
  MonitorElement * eventId_;
  MonitorElement * lumisecId_;
  MonitorElement * timeStamp_;

  /// counters and flags
  int nevt_;
  int nlumisecs_;

  struct{
    timeval startTV,updateTV;
    float startTime;
    float elapsedTime; 
    float updateTime;
  } psTime_;    

  bool saved_;
};

#endif
