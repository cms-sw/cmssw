#ifndef DQMDCSINFO_H
#define DQMDCSINFO_H

/*
 * \file DQMDcsInfo.h
 *
 * $Date: 2012/08/02 07:59:10 $
 * $Revision: 1.3 $
 * \author A.Meyer - DESY
 *
*/

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/Run.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/ParameterSet/interface/Registry.h>
#include <FWCore/ServiceRegistry/interface/Service.h>

#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>

class DQMDcsInfo: public edm::EDAnalyzer{

public:

  /// Constructor
  DQMDcsInfo(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DQMDcsInfo();

protected:

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void beginRun(const edm::Run& r, const edm::EventSetup& c) ;
  void endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c);

private:

  void makeDcsInfo(const edm::Event& e);  
  void makeGtInfo(const edm::Event& e);

  DQMStore *dbe_;

  edm::ParameterSet parameters_;
  std::string subsystemname_;
  std::string dcsinfofolder_;
  
  bool dcs[25];
   // histograms
  MonitorElement * DCSbyLS_ ;
  
};

#endif
