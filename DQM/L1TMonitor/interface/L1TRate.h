#ifndef DQM_L1TMONITOR_L1TRATE_H
#define DQM_L1TMONITOR_L1TRATE_H

/*
 * \file L1TRate.h
 *
 * \author J. Pela
 *
*/

// system include files
#include <memory>
#include <unistd.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/Level1TriggerRates.h"
#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

#include "DQM/L1TMonitor/interface/L1TMenuHelper.h"

#include <TString.h>

#include <iostream>
#include <fstream>
#include <vector>

//
// class declaration
//

class L1TRate : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  L1TRate(const edm::ParameterSet& ps);  // Constructor
  ~L1TRate() override;                   // Destructor

protected:
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;  // Analyze
  //void beginJob();                                                   // BeginJob
  //void endJob  ();                                                   // EndJob
  void bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) override;
  //void dqmEndRun  (const edm::Run& run, const edm::EventSetup& iSetup);

  void beginLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) override;
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;

  // Private methods
private:
  bool getXSexFitsOMDS(const edm::ParameterSet& ps);
  bool getXSexFitsPython(const edm::ParameterSet& ps);

  // Private variables
private:
  // bool
  bool m_verbose;

  // int
  int m_refPrescaleSet;  // What is the reference prescale index to use for trigger choice
  int m_maxNbins;        // Maximum number of bins for MonitorElement
  int m_lsShiftGTRates;  // What shift (if any) to be applied to GT Rates LS number

  // string
  std::string m_outputFile;  // External output file name (for testiting)

  // Vectors
  const std::vector<std::vector<int> >* m_listsPrescaleFactors;  // Collection os all sets of prescales

  // Maps
  std::map<int, int> m_lsPrescaleIndex;                         // Map of precale index for each LS
  std::map<int, double> m_lsLuminosity;                         // Map of luminosity recorded for each LS
  std::map<int, std::map<TString, double> > m_lsRates;          // Map of rates (by bit) recorded for each LS
  std::map<TString, int> m_algoBit;                             // Map of bit associated with a L1 Algo alias
  std::map<std::string, bool> m_inputCategories;                // Map of categories to monitor
  std::map<std::string, std::string> m_selectedTriggers;        // Map of what trigger to monitor for each category
  std::map<TString, MonitorElement*> m_xSecObservedToExpected;  // Monitor Elements for Observed to Expected Algo XSec
  std::map<TString, MonitorElement*> m_xSecVsInstLumi;          // Monitor Elements for Algo XSec vs Instant Luminosity
  std::map<TString, TF1*> m_templateFunctions;                  // For each trigger template f(InstLumi)=XSec

  // Input tags
  edm::EDGetTokenT<LumiScalersCollection> m_scalersSource_colLScal;                 // Where to get L1 Scalers
  edm::EDGetTokenT<Level1TriggerScalersCollection> m_scalersSource_triggerScalers;  // Where to get L1 Scalers
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> m_l1GtDataDaqInputTag;             // Where to get L1 GT Data DAQ
  const edm::ESGetToken<L1GtTriggerMenu, L1GtTriggerMenuRcd> m_menuToken;
  const edm::ESGetToken<L1GtPrescaleFactors, L1GtPrescaleFactorsAlgoTrigRcd> m_l1GtPfAlgoToken;
  L1TMenuHelper::Tokens m_helperTokens;

  // ParameterSet
  edm::ParameterSet m_parameters;

  // MonitorElement
  MonitorElement* m_ErrorMonitor;

  // Others
  //DQMStore* dbe;  // The DQM Service Handle

  L1GtUtils m_l1GtUtils;
};

#endif
