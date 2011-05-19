#ifndef L1TSync_H
#define L1TSync_H

/*
 * \file L1TSync.h
 *
 * $Date: 2011/05/12 13:50:40 $
 * $Revision: 1.3 $
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
#include <string>
#include <vector>

class L1TSync : public edm::EDAnalyzer {

  public:

    L1TSync(const edm::ParameterSet& ps);   // Constructor
    virtual ~L1TSync();                     // Destructor

  protected:

    void analyze(const edm::Event& e, const edm::EventSetup& c); // Analyze
    void beginJob();                                             // BeginJob
    void endJob(void);                                           // EndJob
    void beginRun(const edm::Run& run, const edm::EventSetup& iSetup);
    void endRun  (const edm::Run& run, const edm::EventSetup& iSetup);

    virtual void endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c);

  private:

    DQMStore * dbe; // The DQM Service Handle

    edm::ParameterSet                      m_parameters;

    std::string                            m_outputFile;          // file name for ROOT ouput

    // bool
    bool                                   m_verbose;
    bool*                                  m_processedLS;                   
   
    // Int
    int                                    m_refPrescaleSet;
    unsigned int                            m_currentLS;           // Current Luminosity Section
    unsigned int                            m_eventLS;
    unsigned int                            m_lhcFill;             //

    // Vectors
    std::vector<bool>                      m_bunchStructure;      // Current Bunch Structure

    // Const Vectors
    const std::vector< std::vector<int> >* ListsPrescaleFactors;  // Collection os all sets of prescales

    // Maps
    std::map<TString,int>                  m_algoBit;
    std::map<TString,MonitorElement*>      m_algoCertification;
    std::map<TString,MonitorElement*>      m_algoVsBunchStructure;

    std::map<std::string,bool>             m_algoAutoSelect;      // Map of categories to monitor
    std::map<std::string,std::string>      m_selectedTriggers;    // Map of what trigger to monitor for each category

    // Input tags
    edm::InputTag                          m_scalersSource;       // Where to get L1 Scalers
    edm::InputTag                          m_l1GtEvmSource;
    edm::InputTag                          m_l1GtDataDaqInputTag;

};

#endif
