#ifndef DQMProvInfo_H
#define DQMProvInfo_H

/*
 * \file DQMProvInfo.h
 *
 * $Date: 2009/12/12 13:31:22 $
 * $Revision: 1.6 $
 * \author A.Meyer - DESY
 *
*/

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/Run.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/ServiceRegistry/interface/Service.h>

#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sys/time.h>

class DQMProvInfo: public edm::EDAnalyzer{

public:

  /// Constructor
  DQMProvInfo(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DQMProvInfo();

protected:

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void beginRun(const edm::Run& r, const edm::EventSetup& c) ;
  void endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c);

private:

  std::string getShowTags(void);
  void makeProvInfo();  
  void makeDcsInfo(const edm::Event& e);  
  void makeGtInfo(const edm::Event& e);

  DQMStore *dbe_;

  edm::ParameterSet parameters_;
  
  std::string provinfofolder_;
  std::string subsystemname_;
  int lastlumi_;
  
  bool physDecl_;
  bool dcs24[24];

   // histograms
  MonitorElement * versCMSSW_ ;
  MonitorElement * versDataset_ ;
  MonitorElement * versTaglist_ ;
  MonitorElement * versGlobaltag_ ;
  MonitorElement * hostName_;          ///Hostname of the local machine
  MonitorElement * processName_;       ///DQM "name" of the job (eg, Hcal or DT)
  MonitorElement * workingDir_;        ///Current working directory of the job
  MonitorElement * processId_;         ///The PID associated with this job
  MonitorElement * isComplete_;
  MonitorElement * fileVersion_;
  
  MonitorElement * reportSummary_;
  MonitorElement * reportSummaryMap_;
  
};

#endif
