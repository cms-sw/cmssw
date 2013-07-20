#ifndef DQMOFFLINE_L1TRIGGER_L1TEFFICIENCYHARVESTING_H
#define DQMOFFLINE_L1TRIGGER_L1TEFFICIENCYHARVESTING_H

/*
 * \file L1TEfficiencyHarvesting.h
 *
 * $Date: 2013/03/18 17:17:52 $
 * $Revision: 1.2 $
 * \author J. Pela, C. Battilana
 *
 */

// system include files
#include <memory>
#include <unistd.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TString.h>

#include <iostream>
#include <fstream>
#include <vector>

//
// Efficiency helper class declaration
//

class L1TEfficiencyPlotHandler {

 public:

  L1TEfficiencyPlotHandler(std::string dir, std::string plotName, DQMStore* dbe)  : 
    m_dir(dir), m_plotName(plotName), m_dbe(dbe), m_effHisto(0) { };

  L1TEfficiencyPlotHandler(const L1TEfficiencyPlotHandler &handler);

  ~L1TEfficiencyPlotHandler() { };

  // book efficiency histo
  void book(bool verbose);

  // compute efficiency
  void computeEfficiency(bool verbose);

 private :

  std::string m_dir;
  std::string m_plotName;
  DQMStore * m_dbe;  

  MonitorElement* m_effHisto;

};

//
// DQM class declaration
//

class L1TEfficiency_Harvesting : public edm::EDAnalyzer {
  
public:
  
  L1TEfficiency_Harvesting(const edm::ParameterSet& ps);   // Constructor
  virtual ~L1TEfficiency_Harvesting();                     // Destructor
  
protected:
  
  // Event
  void analyze (const edm::Event& e, const edm::EventSetup& c); 
  
  // Job
  void beginJob();  
  void endJob  ();
  
  // Run
  void beginRun(const edm::Run& run, const edm::EventSetup& iSetup);
  void endRun  (const edm::Run& run, const edm::EventSetup& iSetup);
  
  // Luminosity Block
  virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c);
  virtual void endLuminosityBlock  (edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c);
  
private:

  // bool
  bool  m_verbose;
  
  std::vector<L1TEfficiencyPlotHandler> m_plotHandlers;
  
};

#endif
