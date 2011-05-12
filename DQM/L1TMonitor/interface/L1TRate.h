#ifndef DQM_L1TMONITOR_L1TRATE_H
#define DQM_L1TMONITOR_L1TRATE_H

/*
 * \file L1TRate.h
 *
 * $Date: 2011/04/06 16:49:34 $
 * $Revision: 1.1 $
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

class L1TRate : public edm::EDAnalyzer {

  public:

    L1TRate(const edm::ParameterSet& ps);   // Constructor
    virtual ~L1TRate();                     // Destructor

  protected:

    void analyze (const edm::Event& e, const edm::EventSetup& c);      // Analyze
    void beginJob();                                                   // BeginJob
    void endJob  ();                                                   // EndJob
    void beginRun(const edm::Run& run, const edm::EventSetup& iSetup);
    void endRun  (const edm::Run& run, const edm::EventSetup& iSetup);

    virtual void endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c);

  private:

    // Methods
    int getXSexFitsOMDS  (const edm::ParameterSet& ps); 
    int getXSexFitsPython(const edm::ParameterSet& ps);

    edm::ParameterSet                      m_parameters;

    DQMStore*                              dbe;                      // The DQM Service Handle

    std::string                            m_outputFile;             // file name for ROOT ouput
    //ofstream                             logFile_;                 //

    // bool
    bool                                   m_testEventScalLS;
    bool                                   m_verbose;
    bool*                                  m_processedLS;            // Lists already processed LS

    // Int
    int                                    m_refPrescaleSet;
    int                                    m_maxNbins;
    unsigned int                           m_currentLS;              // Current Luminosity Section

    // Double
    double                                 m_bufferInstLumi;         // 

    // Vectors
    const std::vector< std::vector<int> >* m_listsPrescaleFactors;   // Collection os all sets of prescales
   
    // Maps
    std::map<TString,int>                  m_algoBit;                // Map of bit associated with a L1 Algo alias
    std::map<std::string,bool>             m_inputCategories;        // Map of categories to monitor
    std::map<std::string,std::string>      m_selectedTriggers;       // Map of what trigger to monitor for each category
    std::map<TString,MonitorElement*>      m_xSecObservedToExpected; // Monitor Elements for Observed to Expected Algo XSec 
    std::map<TString,MonitorElement*>      m_xSecVsInstLumi;         // Monitor Elements for Algo XSec vs Instant Luminosity
    std::map<TString,TF1*>                 m_templateFunctions;      // For each trigger template f(InstLumi)=XSec
    std::map<TString,double>               m_bufferRate;

    // Input tags
    edm::InputTag                          m_scalersSource;          // Where to get L1 Scalers
    edm::InputTag                          m_l1GtDataDaqInputTag;    // Where to get L1 GT Data DAQ

};

#endif
