#ifndef EfficiencyPlotter_H
#define EfficiencyPlotter_H


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Run.h"

#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <map>


class EfficiencyPlotter: public edm::EDAnalyzer{

public:

  /// Constructor
  EfficiencyPlotter(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~EfficiencyPlotter();

protected:

  /// BeginJob
  void beginJob(void);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);


private:

  // counters
  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;
  // Switch for verbosity
  std::string metname;

  DQMStore* theDbe;
  edm::ParameterSet parameters;

   //histo binning parameters
  int etaBin;
  double etaMin;
  double etaMax;

  int ptBin;
  double ptMin;
  double ptMax;

  int phiBin;
  double phiMin;
  double phiMax;

  // efficiency histograms
  MonitorElement* ptEfficiency;
  MonitorElement* etaEfficiency;
  MonitorElement* phiEfficiency;

};

#endif
