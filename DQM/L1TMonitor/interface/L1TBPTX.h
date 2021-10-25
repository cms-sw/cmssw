#ifndef L1TBPTX_H
#define L1TBPTX_H

/*
 * \file L1TBPTX.h
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

//Data Formats
#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include <TString.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

class RateBuffer {
public:
  void fill(int ls, bool isAlgo, int bit, double rate) {
    if (isAlgo) {
      m_lsAlgoRate[std::pair<int, int>(ls, bit)] = rate;
    } else {
      m_lsTechRate[std::pair<int, int>(ls, bit)] = rate;
    }
  }
  double getLSRate(int ls, bool isAlgo, int bit, double rate) {
    if (isAlgo) {
      return m_lsAlgoRate[std::pair<int, int>(ls, bit)];
    } else {
      return m_lsTechRate[std::pair<int, int>(ls, bit)];
    }
  }
  double getLSAlgoRate(int ls, int bit, double rate) { return m_lsAlgoRate[std::pair<int, int>(ls, bit)]; }
  double getLSTechRate(int ls, int bit, double rate) { return m_lsTechRate[std::pair<int, int>(ls, bit)]; }

private:
  std::map<std::pair<int, int>, double> m_lsAlgoRate;
  std::map<std::pair<int, int>, double> m_lsTechRate;
};

class L1GtTriggerMenu;
class L1GtTriggerMenuRcd;
class L1GtPrescaleFactors;
class L1GtPrescaleFactorsAlgoTrigRcd;
class L1GtPrescaleFactorsTechTrigRcd;

class L1TBPTX : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
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
    SQUEEZE = 9,
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
  L1TBPTX(const edm::ParameterSet& ps);  // Constructor
  ~L1TBPTX() override;                   // Destructor

protected:
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;  // Analyze
  void bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) override;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

  void beginLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) override;

  // Private Methods
private:
  void getBeamConfOMDS();
  void doFractionInSync(bool iForce = false, bool iBad = false);
  void certifyLSBlock(std::string iTrigger, int iInitLs, int iEndLs, float iValue);

  // Variables
private:
  edm::ParameterSet m_parameters;
  std::vector<edm::ParameterSet> m_monitorBits;
  std::vector<edm::ParameterSet> m_monitorRates;
  std::string m_outputFile;  // file name for ROOT ouput

  // bool
  bool m_verbose;
  bool m_currentLSValid;
  bool* m_processedLS;

  // Int
  std::map<TString, int> m_effNumerator;
  std::map<TString, int> m_effDenominator;
  std::map<TString, int> m_missFireNumerator;
  std::map<TString, int> m_missFireDenominator;

  int m_refPrescaleSet;
  int m_currentPrescalesIndex;
  unsigned int m_currentLS;  // Current LS
  unsigned int m_currentGTLS;
  //unsigned int                         m_eventLS;
  unsigned int m_lhcFill;  //

  // Vectors
  BeamConfiguration m_beamConfig;  // Current Bunch Structure
  std::vector<std::pair<int, int> > m_selAlgoBit;
  std::vector<std::pair<int, int> > m_selTechBit;

  // Const Vectors
  const std::vector<std::vector<int> >* m_prescaleFactorsAlgoTrig;
  const std::vector<std::vector<int> >* m_prescaleFactorsTechTrig;

  // Maps
  std::map<int, TString> m_algoBit_Alias;
  std::map<int, TString> m_techBit_Alias;

  std::map<TString, MonitorElement*> m_meAlgoEfficiency;
  std::map<TString, MonitorElement*> m_meAlgoMissFire;
  std::map<TString, MonitorElement*> m_meTechEfficiency;
  std::map<TString, MonitorElement*> m_meTechMissFire;

  std::map<std::pair<bool, int>, MonitorElement*> m_meRate;
  std::map<std::pair<bool, int>, double> m_l1Rate;

  // MonitorElement
  MonitorElement* m_ErrorMonitor;

  // Input tags
  edm::EDGetTokenT<Level1TriggerScalersCollection> m_scalersSource;  // Where to get L1 Scalers
  edm::EDGetTokenT<L1GlobalTriggerEvmReadoutRecord> m_l1GtEvmSource;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> m_l1GtDataDaqInputTag;
  edm::ESGetToken<L1GtTriggerMenu, L1GtTriggerMenuRcd> l1gtMenuToken_;
  edm::ESGetToken<L1GtPrescaleFactors, L1GtPrescaleFactorsAlgoTrigRcd> l1GtPfAlgoToken_;
  edm::ESGetToken<L1GtPrescaleFactors, L1GtPrescaleFactorsTechTrigRcd> l1GtPfTechToken_;
};

#endif
