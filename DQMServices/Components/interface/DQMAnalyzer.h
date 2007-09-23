#ifndef DQMAnalyzer_H
#define DQMAnalyzer_H

/*
 * \file DQMAnalyzer.h
 *
 * $Date: 2007/03/30 13:57:00 $
 * $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
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



class DQMAnalyzer: public edm::EDAnalyzer{

public:

  /// Constructors
  
  DQMAnalyzer(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DQMAnalyzer();

protected:

  /// to be used by derived class
  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// BeginRun
  void beginRun(const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) ;

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                            const edm::EventSetup& c) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);

  /// EndRun
  void endRun(const edm::Run& run, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  /// Save DQM output file
  void save(std::string flag="");

  DaqMonitorBEInterface* dbe;
  
  edm::ParameterSet parameters;
  int PSprescale ;
  std::string PSrootFolder ;

  int getNumEvents(){return nevt_;}
  int getNumLumiSecs(){return nlumisecs_;}

  // FIXME: put additional methods here
  // processME();

private:

  /// environment
  int irun_,ilumisec_,ievent_,itime_;

  /// framework ME
  MonitorElement * runId_;
  MonitorElement * eventId_;
  MonitorElement * lumisecId_;
  MonitorElement * timeStamp_;

  /// counters and flags
  int nevt_ ;
  int nlumisecs_ ;
  bool saved_ ;
  bool debug_ ;
// FIXME  bool actonLS_ ;

};

#endif
