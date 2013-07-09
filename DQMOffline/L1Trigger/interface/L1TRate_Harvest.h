#ifndef DQMOFFLINE_L1TRIGGER_L1TRATE_HARVEST_H
#define DQMOFFLINE_L1TRIGGER_L1TRATE_HARVEST_H

/*
 * \file L1TRate_Harvest.h
 *
 * $Date: 2011/10/28 13:24:48 $
 * $Revision: 1.4 $
 * \author J. Pela
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
// class declaration
//

class L1TRate_Harvest : public edm::EDAnalyzer {

 public:

  enum Errors{
    UNKNOWN                = 1,
    WARNING_PY_MISSING_FIT = 2
  };

 public:

    L1TRate_Harvest(const edm::ParameterSet& ps);   // Constructor
    virtual ~L1TRate_Harvest();                     // Destructor

 protected:

    void analyze (const edm::Event& e, const edm::EventSetup& c);      // Analyze
    void beginJob();                                                   // BeginJob
    void endJob  ();                                                   // EndJob
    void beginRun(const edm::Run& run, const edm::EventSetup& iSetup);
    void endRun  (const edm::Run& run, const edm::EventSetup& iSetup);

    virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c);
    virtual void endLuminosityBlock  (edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c);

  // Private methods
  private:

    bool getXSexFitsOMDS  (const edm::ParameterSet& ps); 
    bool getXSexFitsPython(const edm::ParameterSet& ps);

  // Private variables
  private:

    // bool
    bool  m_verbose;
    
    // int
    int  m_refPrescaleSet; // What is the reference prescale index to use for trigger choice
    int  m_maxNbins;       // Maximum number of bins for MonitorElement
    int  m_lsShiftGTRates; // What shift (if any) to be applied to GT Rates LS number

    // string
    std::string  m_outputFile; // External output file name (for testiting)
    
    // Vectors
    const std::vector< std::vector<int> >* m_listsPrescaleFactors; // Collection os all sets of prescales

    // Maps
    std::map<int,int>                       m_lsPrescaleIndex;        // Map of precale index for each LS
    std::map<int,double>                    m_lsLuminosity;           // Map of luminosity recorded for each LS
    std::map<int,std::map<TString,double> > m_lsRates;                // Map of rates (by bit) recorded for each LS
    std::map<TString,int>                   m_algoBit;                // Map of bit associated with a L1 Algo alias
    std::map<std::string,bool>              m_inputCategories;        // Map of categories to monitor
    std::map<std::string,std::string>       m_selectedTriggers;       // Map of what trigger to monitor for each category
    std::map<TString,MonitorElement*>       m_xSecObservedToExpected; // Monitor Elements for Observed to Expected Algo XSec 
    std::map<TString,MonitorElement*>       m_xSecVsInstLumi;         // Monitor Elements for Algo XSec vs Instant Luminosity
    std::map<TString,TF1*>                  m_templateFunctions;      // For each trigger template f(InstLumi)=XSec

    // Input tags
    edm::InputTag m_scalersSource;       // Where to get L1 Scalers
    edm::InputTag m_l1GtDataDaqInputTag; // Where to get L1 GT Data DAQ

    // ParameterSet
    edm::ParameterSet m_parameters;
    
    // MonitorElement
    MonitorElement* m_ErrorMonitor;
    
    // Others
    DQMStore* dbe;  // The DQM Service Handle

};

#endif
