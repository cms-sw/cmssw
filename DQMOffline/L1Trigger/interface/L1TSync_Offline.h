#ifndef DQMOffline_L1Trigger_L1TSync_Offline_h
#define DQMOffline_L1Trigger_L1TSync_Offline_h

/*
 * \class L1TSync_Offline
 *
 *
 * Description: offline DQM module for L1Trigger/bunchStructure synchronization
 * 
 * Implementation:
 *   <TODO: enter implementation details>
 *
 * \author: Pietro Vischia - LIP Lisbon pietro.vischia@gmail.com
 *
 * Changelog:
 *    2012/08/10 11:01:01: First creation. Dummy module with actual code commented.
 *
 * Todo:
 *  - implement the module in offline
 *  - check if there are user includes specific for offline/online that should be changed
 *
 *
 */

// System include files
#include <memory>
#include <unistd.h>

// User include files
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

//L1 includes and dataformats
#include "DQMOffline/L1Trigger/interface/L1TBeamConfiguration.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

#include <TString.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// Forward declarations

// Class declaration
class L1TSync_Offline : public DQMEDAnalyzer {

  public:

    enum BeamMode{
      NOMODE=1,
      SETUP=2,
      INJPILOT=3,
      INJINTR=4,
      INJNOMN=5,
      PRERAMP=6,
      RAMP=7,
      FLATTOP=8,
      QUEEZE=9,
      ADJUST=10,
      STABLE=11,
      UNSTABLE=12,
      BEAMDUMP=13,
      RAMPDOWN=14,
      RECOVERY=15,
      INJDUMP=16,
      CIRCDUMP=17,
      ABORT=18,
      CYCLING=19,
      WBDUMP=20,
      NOBEAM=21
    };

    enum Errors{
      UNKNOWN                       = 1,
      WARNING_DB_CONN_FAILED        = 2,
      WARNING_DB_QUERY_FAILED       = 3,
      WARNING_DB_INCORRECT_NBUNCHES = 4,
      ERROR_UNABLE_RETRIVE_PRODUCT  = 5,
      ERROR_TRIGGERALIAS_NOTVALID   = 6,
      ERROR_LSBLOCK_NOTVALID        = 7
    };


 public:
    // Constructor
    L1TSync_Offline(const edm::ParameterSet& ps);   
    // Destructor
    virtual ~L1TSync_Offline();                     
    
  protected:

  void analyze (const edm::Event& e, const edm::EventSetup& c);  // Analyze
  virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c);
  virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&);
  virtual void bookHistograms(DQMStore::IBooker &ibooker, const edm::Run&, const edm::EventSetup&) override;
// no lumi block //    virtual void endLuminosityBlock  (edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c);


  // Private Methods
  private:

    void getBeamConfOffline(const edm::Event&);
// no lumi block //    void doFractionInSync(bool iForce=false, bool iBad=false);
// no lumi block //    void certifyLSBlock  (std::string iTrigger, int iInitLs, int iEndLs ,float iValue);

  // Variables
  private:

    // Input parameters
    edm::ParameterSet                      m_parameters;

    // ROOT output filename
    std::string                            m_outputFile;        

    // bool
    bool                                   m_verbose;
    bool                                   m_currentLSValid;
    bool*                                  m_processedLS;                      

    // Int
    int                                    m_refPrescaleSet;
    unsigned int                           m_currentLS;            // Current LS
    //unsigned int                         m_eventLS;
    unsigned int                           m_lhcFill;             //

    // Vectors
    L1TBeamConfiguration                      m_beamConfig;          // Current Bunch Structure

    // Const Vectors
    const std::vector< std::vector<int> >* ListsPrescaleFactors;  // Collection os all sets of prescales

    // Maps
    std::map<TString,int>                  m_algoBit;
    std::map<TString,unsigned int>         m_certFirstLS;         // First uncertified LS
    std::map<TString,unsigned int>         m_certLastLS;          // Last  uncertified LS
    std::map<TString,MonitorElement*>      m_algoCertification;
    std::map<TString,MonitorElement*>      m_algoVsBunchStructure;
    std::map<std::string,bool>             m_algoAutoSelect;      // Map of categories to monitor
    std::map<std::string,std::string>      m_selectedTriggers;    // Map of what trigger to monitor for each category

    // MonitorElement
    MonitorElement*                        m_ErrorMonitor;

    // Input tags
    edm::EDGetTokenT<L1GlobalTriggerEvmReadoutRecord> m_l1GtEvmSource;
    edm::EDGetTokenT<L1GlobalTriggerReadoutRecord>    m_l1GtDataDaqInputTag;

    L1GtUtils m_l1GtUtils;
};

#endif
