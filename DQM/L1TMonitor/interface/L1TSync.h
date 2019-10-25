#ifndef L1TSync_H
#define L1TSync_H

/*
 * \file L1TSync.h
 *
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
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/L1TMonitor/interface/L1TOMDSHelper.h"

//DataFormats
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

#include <TString.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

class L1TSync : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  enum BeamMode {
    NOMODE = 1,
    SETUP = 2,
    INJPILOT = 3,
    INJINTR = 4,
    INJNOMN = 5,
    PRERAMP = 6,
    RAMP = 7,
    FLATTOP = 8,
    QUEEZE = 9,
    ADJUST = 10,
    STABLE = 11,
    UNSTABLE = 12,
    BEAMDUMP = 13,
    RAMPDOWN = 14,
    RECOVERY = 15,
    INJDUMP = 16,
    CIRCDUMP = 17,
    ABORT = 18,
    CYCLING = 19,
    WBDUMP = 20,
    NOBEAM = 21
  };

  enum Errors {
    UNKNOWN = 1,
    WARNING_DB_CONN_FAILED = 2,
    WARNING_DB_QUERY_FAILED = 3,
    WARNING_DB_INCORRECT_NBUNCHES = 4,
    ERROR_UNABLE_RETRIVE_PRODUCT = 5,
    ERROR_TRIGGERALIAS_NOTVALID = 6,
    ERROR_LSBLOCK_NOTVALID = 7
  };

public:
  L1TSync(const edm::ParameterSet& ps);  // Constructor
  ~L1TSync() override;                   // Destructor

protected:
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;  // Analyze

  void beginLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) override;
  void bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) override;
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;

  // Private Methods
private:
  void getBeamConfOMDS();
  void doFractionInSync(bool iForce = false, bool iBad = false);
  void certifyLSBlock(std::string iTrigger, int iInitLs, int iEndLs, float iValue);

  // Variables
private:
  edm::ParameterSet m_parameters;

  std::string m_outputFile;  // file name for ROOT ouput

  // bool
  bool m_verbose;
  bool m_currentLSValid;
  bool* m_processedLS;

  // Int
  int m_refPrescaleSet;
  unsigned int m_currentLS;  // Current LS
  //unsigned int                         m_eventLS;
  unsigned int m_lhcFill;  //

  // Vectors
  BeamConfiguration m_beamConfig;  // Current Bunch Structure

  // Const Vectors
  const std::vector<std::vector<int> >* ListsPrescaleFactors;  // Collection os all sets of prescales

  // Maps
  std::map<TString, int> m_algoBit;
  std::map<TString, unsigned int> m_certFirstLS;  // First uncertified LS
  std::map<TString, unsigned int> m_certLastLS;   // Last  uncertified LS
  std::map<TString, MonitorElement*> m_algoCertification;
  std::map<TString, MonitorElement*> m_algoVsBunchStructure;
  std::map<std::string, bool> m_algoAutoSelect;           // Map of categories to monitor
  std::map<std::string, std::string> m_selectedTriggers;  // Map of what trigger to monitor for each category

  // MonitorElement
  MonitorElement* m_ErrorMonitor;

  // Input tags
  edm::EDGetTokenT<L1GlobalTriggerEvmReadoutRecord> m_l1GtEvmSource;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> m_l1GtDataDaqInputTag;

  L1GtUtils m_l1GtUtils;
};

#endif
