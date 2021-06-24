#include "DQM/L1TMonitor/interface/L1TBPTX.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/Level1TriggerRates.h"

#include "DataFormats/Common/interface/ConditionsInEdm.h"  // Parameters associated to Run, LS and Event

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"

// Luminosity Information
//#include "DataFormats/Luminosity/interface/LumiDetails.h"
//#include "DataFormats/Luminosity/interface/LumiSummary.h"

// L1TMonitor includes
#include "DQM/L1TMonitor/interface/L1TMenuHelper.h"

#include "TList.h"
#include <string>

using namespace edm;
using namespace std;

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
L1TBPTX::L1TBPTX(const ParameterSet& pset) {
  m_parameters = pset;

  // Mapping parameter input variables
  m_scalersSource = consumes<Level1TriggerScalersCollection>(pset.getParameter<InputTag>("inputTagScalersResults"));
  m_l1GtDataDaqInputTag = consumes<L1GlobalTriggerReadoutRecord>(pset.getParameter<InputTag>("inputTagL1GtDataDaq"));
  m_l1GtEvmSource = consumes<L1GlobalTriggerEvmReadoutRecord>(pset.getParameter<InputTag>("inputTagtEvmSource"));
  l1gtMenuToken_ = esConsumes<edm::Transition::BeginRun>();
  l1GtPfAlgoToken_ = esConsumes<edm::Transition::BeginRun>();
  l1GtPfTechToken_ = esConsumes<edm::Transition::BeginRun>();
  m_verbose = pset.getUntrackedParameter<bool>("verbose", false);
  //  m_refPrescaleSet      = pset.getParameter         <int>     ("refPrescaleSet");

  m_monitorBits = pset.getParameter<vector<ParameterSet> >("MonitorBits");

  for (unsigned i = 0; i < m_monitorBits.size(); i++) {
    // Algorithms
    if (m_monitorBits[i].getParameter<bool>("bitType")) {
      int bit = m_monitorBits[i].getParameter<int>("bitNumber");
      int offset = m_monitorBits[i].getParameter<int>("bitOffset");
      m_selAlgoBit.push_back(pair<int, int>(bit, offset));
    }
    // Tech
    else {
      int bit = m_monitorBits[i].getParameter<int>("bitNumber");
      int offset = m_monitorBits[i].getParameter<int>("bitOffset");
      m_selTechBit.push_back(pair<int, int>(bit, offset));
    }
  }

  m_monitorRates = pset.getParameter<vector<ParameterSet> >("MonitorRates");
}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
L1TBPTX::~L1TBPTX() {}

//-------------------------------------------------------------------------------------
/// BeginRun
//-------------------------------------------------------------------------------------
void L1TBPTX::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run& iRun, const edm::EventSetup& iSetup) {
  if (m_verbose) {
    cout << "[L1TBPTX] Called beginRun." << endl;
  }

  ibooker.setCurrentFolder("L1T/L1TBPTX");

  // Initializing variables
  int maxNbins = 2501;

  // Reseting run dependent variables
  m_lhcFill = 0;
  m_currentLS = 0;

  // Getting Trigger menu from GT
  const L1GtTriggerMenu* menu = &iSetup.getData(l1gtMenuToken_);

  // Filling Alias-Bit Map
  for (CItAlgo algo = menu->gtAlgorithmAliasMap().begin(); algo != menu->gtAlgorithmAliasMap().end(); ++algo) {
    m_algoBit_Alias[(algo->second).algoBitNumber()] = (algo->second).algoAlias();
  }

  for (CItAlgo algo = menu->gtTechnicalTriggerMap().begin(); algo != menu->gtTechnicalTriggerMap().end(); ++algo) {
    m_techBit_Alias[(algo->second).algoBitNumber()] = (algo->second).algoName();
  }

  // Initializing DQM Monitor Elements
  ibooker.setCurrentFolder("L1T/L1TBPTX");
  m_ErrorMonitor = ibooker.book1D("ErrorMonitor", "ErrorMonitor", 7, 0, 7);
  m_ErrorMonitor->setBinLabel(UNKNOWN, "UNKNOWN");
  m_ErrorMonitor->setBinLabel(WARNING_DB_CONN_FAILED, "WARNING_DB_CONN_FAILED");    // Errors from L1TOMDSHelper
  m_ErrorMonitor->setBinLabel(WARNING_DB_QUERY_FAILED, "WARNING_DB_QUERY_FAILED");  // Errors from L1TOMDSHelper
  m_ErrorMonitor->setBinLabel(WARNING_DB_INCORRECT_NBUNCHES,
                              "WARNING_DB_INCORRECT_NBUNCHES");  // Errors from L1TOMDSHelper
  m_ErrorMonitor->setBinLabel(ERROR_UNABLE_RETRIVE_PRODUCT, "ERROR_UNABLE_RETRIVE_PRODUCT");
  m_ErrorMonitor->setBinLabel(ERROR_TRIGGERALIAS_NOTVALID, "ERROR_TRIGGERALIAS_NOTVALID");
  m_ErrorMonitor->setBinLabel(ERROR_LSBLOCK_NOTVALID, "ERROR_LSBLOCK_NOTVALID");

  for (unsigned i = 0; i < m_monitorBits.size(); i++) {
    bool isAlgo = m_monitorBits[i].getParameter<bool>("bitType");
    TString testName = m_monitorBits[i].getParameter<string>("testName");
    int bit = m_monitorBits[i].getParameter<int>("bitNumber");

    TString meTitle = "";
    ibooker.setCurrentFolder("L1T/L1TBPTX/Efficiency/");
    if (isAlgo) {
      meTitle = "Algo ";
      meTitle += bit;
      meTitle += " - ";
      meTitle += m_algoBit_Alias[bit];
      meTitle += " Efficiency";
      m_meAlgoEfficiency[bit] = ibooker.book1D(testName, meTitle, maxNbins, -0.5, double(maxNbins) - 0.5);
      m_meAlgoEfficiency[bit]->setAxisTitle("Lumi Section", 1);
    } else {
      meTitle = "Tech ";
      meTitle += bit;
      meTitle += " - ";
      meTitle += m_techBit_Alias[bit];
      meTitle += " Efficiency";
      m_meTechEfficiency[bit] = ibooker.book1D(testName, meTitle, maxNbins, -0.5, double(maxNbins) - 0.5);
      m_meTechEfficiency[bit]->setAxisTitle("Lumi Section", 1);
    }

    meTitle = "";
    ibooker.setCurrentFolder("L1T/L1TBPTX/MissFire/");
    if (isAlgo) {
      meTitle = "Algo ";
      meTitle += bit;
      meTitle += " - ";
      meTitle += m_algoBit_Alias[bit];
      meTitle += "(1 - Miss Fire Rate)";
      m_meAlgoMissFire[bit] = ibooker.book1D(testName, meTitle, maxNbins, -0.5, double(maxNbins) - 0.5);
      m_meAlgoMissFire[bit]->setAxisTitle("Lumi Section", 1);
      m_meAlgoMissFire[bit]->setAxisTitle("1 - Miss Fire Rate", 2);
    } else {
      meTitle = "Tech ";
      meTitle += bit;
      meTitle += " - ";
      meTitle += m_techBit_Alias[bit];
      meTitle += "(1 - Miss Fire Rate)";
      m_meTechMissFire[bit] = ibooker.book1D(testName, meTitle, maxNbins, -0.5, double(maxNbins) - 0.5);
      m_meTechMissFire[bit]->setAxisTitle("Lumi Section", 1);
      m_meTechMissFire[bit]->setAxisTitle("1 - Miss Fire Rate", 2);
    }
  }

  for (unsigned i = 0; i < m_monitorRates.size(); i++) {
    TString testName = m_monitorRates[i].getParameter<string>("testName");
    bool isAlgo = m_monitorRates[i].getParameter<bool>("bitType");
    int bit = m_monitorRates[i].getParameter<int>("bitNumber");

    pair<bool, int> refME = pair<bool, int>(isAlgo, bit);

    TString meTitle = "";
    ibooker.setCurrentFolder("L1T/L1TBPTX/Rate/");
    if (isAlgo) {
      meTitle = "Algo " + std::to_string(bit);
      meTitle += " - ";
      meTitle += m_algoBit_Alias[bit];
      meTitle += " Rate";
      m_meRate[refME] = ibooker.book1D(testName, meTitle, maxNbins, -0.5, double(maxNbins) - 0.5);
      m_meRate[refME]->setAxisTitle("Lumi Section", 1);
      m_meRate[refME]->setAxisTitle("Rate (unprescaled) [Hz]", 2);
    } else {
      meTitle = "Tech " + std::to_string(bit);
      meTitle += " - ";
      meTitle += m_techBit_Alias[bit];
      meTitle += " Rate";
      m_meRate[refME] = ibooker.book1D(testName, meTitle, maxNbins, -0.5, double(maxNbins) - 0.5);
      m_meRate[refME]->setAxisTitle("Lumi Section", 1);
      m_meRate[refME]->setAxisTitle("Rate (unprescaled) [Hz]", 2);
    }
  }

  //_____________________________________________________________________
  // Getting the prescale columns definition for this run
  const auto& l1GtPfAlgo = iSetup.getHandle(l1GtPfAlgoToken_);
  const auto& l1GtPfTech = iSetup.getHandle(l1GtPfTechToken_);

  if (l1GtPfAlgo.isValid()) {
    const L1GtPrescaleFactors* m_l1GtPfAlgo = l1GtPfAlgo.product();
    m_prescaleFactorsAlgoTrig = &(m_l1GtPfAlgo->gtPrescaleFactors());
  } else {
    //TODO: Some error handling
  }

  if (l1GtPfAlgo.isValid()) {
    const L1GtPrescaleFactors* m_l1GtPfTech = l1GtPfTech.product();
    m_prescaleFactorsTechTrig = &(m_l1GtPfTech->gtPrescaleFactors());
  } else {
    //TODO: Some error handling
  }
}

void L1TBPTX::dqmBeginRun(const edm::Run&, const edm::EventSetup&) {
  //empty
}

//_____________________________________________________________________
// Function: beginLuminosityBlock
//_____________________________________________________________________
void L1TBPTX::beginLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {
  if (m_verbose) {
    cout << "[L1TBPTX] Called beginLuminosityBlock." << endl;
  }

  // Updating current LS number
  m_currentLS = lumiBlock.id().luminosityBlock();

  // A LS will be valid if BeamMode==STABLE for all events monitored
  m_currentLSValid = true;

  for (unsigned i = 0; i < m_monitorBits.size(); i++) {
    TString triggerName = "";
    if (m_monitorBits[i].getParameter<bool>("bitType")) {
      triggerName = "algo_" + std::to_string(m_monitorBits[i].getParameter<int>("bitNumber"));
    } else {
      triggerName = "tech_" + std::to_string(m_monitorBits[i].getParameter<int>("bitNumber"));
    }

    m_effNumerator[triggerName] = 0;
    m_effDenominator[triggerName] = 0;
    m_missFireNumerator[triggerName] = 0;
    m_missFireDenominator[triggerName] = 0;
  }
}

//_____________________________________________________________________
// Function: endLuminosityBlock
// * Fills LS by LS ration of trigger out of sync
//_____________________________________________________________________
void L1TBPTX::endLuminosityBlock(LuminosityBlock const& lumiBlock, EventSetup const& c) {
  //______________________________________________________________________________
  // Monitoring efficiencies
  //______________________________________________________________________________
  if (m_verbose) {
    cout << "[L1TBPTX] Called endLuminosityBlock." << endl;
  }

  // If this LS is valid (i.e. all events recorded with stable beams)
  if (m_currentLSValid && m_beamConfig.isValid()) {
    for (unsigned i = 0; i < m_monitorBits.size(); i++) {
      bool isAlgo = m_monitorBits[i].getParameter<bool>("bitType");
      TString testName = m_monitorBits[i].getParameter<string>("testName");
      int bit = m_monitorBits[i].getParameter<int>("bitNumber");

      TString triggerName;
      if (isAlgo) {
        triggerName = "algo_" + std::to_string(bit);
      } else {
        triggerName = "tech_" + std::to_string(bit);
      }

      double valEff;
      double valMiss;
      if (m_effDenominator[triggerName] != 0) {
        valEff = (double)m_effNumerator[triggerName] / m_effDenominator[triggerName];
      } else {
        valEff = 0;
      }
      if (m_missFireDenominator[triggerName] != 0) {
        valMiss = (double)m_missFireNumerator[triggerName] / m_missFireDenominator[triggerName];
      } else {
        valMiss = 0;
      }

      if (isAlgo) {
        int bin = m_meAlgoEfficiency[bit]->getTH1()->FindBin(m_currentLS);
        m_meAlgoEfficiency[bit]->setBinContent(bin, valEff);
        m_meAlgoMissFire[bit]->setBinContent(bin, 1 - valMiss);
      } else {
        int bin = m_meTechEfficiency[bit]->getTH1()->FindBin(m_currentLS);
        m_meTechEfficiency[bit]->setBinContent(bin, valEff);
        m_meTechMissFire[bit]->setBinContent(bin, 1 - valMiss);
      }
    }
  }

  //______________________________________________________________________________
  // Monitoring rates
  //______________________________________________________________________________
  // We are only interested in monitoring lumisections where the the LHC state is
  // RAMP, FLATTOP, SQUEEZE, ADJUST or STABLE since the bunch configuration and
  // therefore the BPTX rate will not change.

  if (m_currentLSValid) {
    const vector<int>& currentPFAlgo = (*m_prescaleFactorsAlgoTrig).at(m_currentPrescalesIndex);
    const vector<int>& currentPFTech = (*m_prescaleFactorsTechTrig).at(m_currentPrescalesIndex);

    for (unsigned i = 0; i < m_monitorRates.size(); i++) {
      bool isAlgo = m_monitorRates[i].getParameter<bool>("bitType");
      int bit = m_monitorRates[i].getParameter<int>("bitNumber");

      pair<bool, int> refME = pair<bool, int>(isAlgo, bit);

      if (isAlgo) {
        int bin = m_meRate[refME]->getTH1()->FindBin(m_currentGTLS);
        int trigPS = currentPFAlgo[bit];
        double trigRate = (double)trigPS * m_l1Rate[refME];
        m_meRate[refME]->setBinContent(bin, trigRate);

      } else {
        int bin = m_meRate[refME]->getTH1()->FindBin(m_currentGTLS);
        int trigPS = currentPFTech[bit];
        double trigRate = (double)trigPS * m_l1Rate[refME];
        m_meRate[refME]->setBinContent(bin, trigRate);
      }
    }
  }
}

//_____________________________________________________________________
void L1TBPTX::analyze(const Event& iEvent, const EventSetup& eventSetup) {
  if (m_verbose) {
    cout << "[L1TBPTX] Called analyze." << endl;
  }

  // We only start analyzing if current LS is still valid
  if (m_currentLSValid) {
    if (m_verbose) {
      cout << "[L1TBPTX] -> m_currentLSValid=" << m_currentLSValid << endl;
    }

    // Retriving information from GT
    edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtEvmReadoutRecord;
    iEvent.getByToken(m_l1GtEvmSource, gtEvmReadoutRecord);

    // Determining beam mode and fill number
    if (gtEvmReadoutRecord.isValid()) {
      const L1GtfeExtWord& gtfeEvmWord = gtEvmReadoutRecord->gtfeWord();
      unsigned int lhcBeamMode = gtfeEvmWord.beamMode();  // Updating beam mode

      if (m_verbose) {
        cout << "[L1TBPTX] Beam mode: " << lhcBeamMode << endl;
      }

      if (lhcBeamMode == RAMP || lhcBeamMode == FLATTOP || lhcBeamMode == SQUEEZE || lhcBeamMode == ADJUST ||
          lhcBeamMode == STABLE) {
        if (m_lhcFill == 0) {
          if (m_verbose) {
            cout << "[L1TBPTX] No valid bunch structure yet retrived. Attemptting to retrive..." << endl;
          }

          m_lhcFill = gtfeEvmWord.lhcFillNumber();  // Getting LHC Fill Number from GT

          getBeamConfOMDS();  // Getting Beam Configuration from OMDS

          // We are between RAMP and STABLE so there should be some colliding bunches
          // in the machine. If 0 colliding bunched are found might be due to a delay
          // of the update of the database. So we declare this LS as invalid and try
          // again on the next one.
          if (m_beamConfig.nCollidingBunches <= 0) {
            m_lhcFill = 0;
            m_currentLSValid = false;
          }
        }
      } else {
        m_currentLSValid = false;
      }

    } else {
      int eCount = m_ErrorMonitor->getTH1()->GetBinContent(ERROR_UNABLE_RETRIVE_PRODUCT);
      eCount++;
      m_ErrorMonitor->getTH1()->SetBinContent(ERROR_UNABLE_RETRIVE_PRODUCT, eCount);
    }
  }

  //______________________________________________________________________________
  // If current LS is valid and Beam Configuration is Valid we analyse this event
  //______________________________________________________________________________
  if (m_currentLSValid && m_beamConfig.isValid()) {
    if (m_verbose) {
      cout << "Current event in valid LS and beam config" << endl;
    }

    // Getting Final Decision Logic (FDL) Data from GT
    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecordData;
    iEvent.getByToken(m_l1GtDataDaqInputTag, gtReadoutRecordData);

    if (gtReadoutRecordData.isValid()) {
      const vector<L1GtFdlWord>& gtFdlVectorData = gtReadoutRecordData->gtFdlVector();

      // Getting the index for the fdl data for this event
      int eventFDL = 0;
      for (unsigned int i = 0; i < gtFdlVectorData.size(); i++) {
        if (gtFdlVectorData[i].bxInEvent() == 0) {
          eventFDL = i;
          break;
        }
      }

      m_currentPrescalesIndex = gtFdlVectorData[eventFDL].gtPrescaleFactorIndexAlgo();

      for (unsigned i = 0; i < m_monitorBits.size(); i++) {
        TString triggerName = "";
        bool isAlgo = m_monitorBits[i].getParameter<bool>("bitType");
        int bit = m_monitorBits[i].getParameter<int>("bitNumber");
        int offset = m_monitorBits[i].getParameter<int>("bitOffset");

        if (isAlgo) {
          triggerName = "algo_" + std::to_string(bit);
        } else {
          triggerName = "tech_" + std::to_string(bit);
        }

        int evBxStart = -2;
        int evBxEnd = 2;

        if (offset < 0) {
          evBxStart += -1 * offset;
        }
        if (offset > 0) {
          evBxEnd += -1 * offset;
        }

        for (unsigned a = 0; a < gtFdlVectorData.size(); a++) {
          int testBx = gtFdlVectorData[a].localBxNr() - offset;
          bool lhcBxFilled = m_beamConfig.beam1[testBx] && m_beamConfig.beam2[testBx];
          bool algoFired = false;

          if (isAlgo) {
            if (gtFdlVectorData[a].gtDecisionWord()[bit]) {
              algoFired = true;
            }

          } else {
            if (gtFdlVectorData[a].gtTechnicalTriggerWord()[bit]) {
              algoFired = true;
            }
          }

          if (lhcBxFilled) {
            m_effDenominator[triggerName]++;
          }
          if (lhcBxFilled && algoFired) {
            m_effNumerator[triggerName]++;
          }

          if (algoFired) {
            m_missFireNumerator[triggerName]++;
          }
          if (algoFired && !lhcBxFilled) {
            m_missFireNumerator[triggerName]++;
          }
        }
      }
    }
  }

  //______________________________________________________________________________
  // Rate calculation
  //______________________________________________________________________________
  edm::Handle<Level1TriggerScalersCollection> triggerScalers;
  iEvent.getByToken(m_scalersSource, triggerScalers);

  if (triggerScalers.isValid()) {
    Level1TriggerScalersCollection::const_iterator itL1TScalers = triggerScalers->begin();
    Level1TriggerRates trigRates(*itL1TScalers, iEvent.id().run());

    m_currentGTLS = (*itL1TScalers).lumiSegmentNr();

    for (unsigned i = 0; i < m_monitorRates.size(); i++) {
      bool isAlgo = m_monitorRates[i].getParameter<bool>("bitType");
      int bit = m_monitorRates[i].getParameter<int>("bitNumber");

      pair<bool, int> refTrig = pair<bool, int>(isAlgo, bit);

      if (isAlgo) {
        m_l1Rate[refTrig] = trigRates.gtAlgoCountsRate()[bit];
      } else {
        m_l1Rate[refTrig] = trigRates.gtTechCountsRate()[bit];
      }
    }
  }
}

//_____________________________________________________________________
// Method: getBunchStructureOMDS
// Description: Attempt to retrive Beam Configuration from OMDS and if
//              we find error handle it
//_____________________________________________________________________
void L1TBPTX::getBeamConfOMDS() {
  if (m_verbose) {
    cout << "[L1TBPTX] Called getBeamConfOMDS()" << endl;
  }

  //Getting connection paremeters
  string oracleDB = m_parameters.getParameter<string>("oracleDB");
  string pathCondDB = m_parameters.getParameter<string>("pathCondDB");

  // Connecting to OMDS
  L1TOMDSHelper myOMDSHelper = L1TOMDSHelper();
  int conError;
  myOMDSHelper.connect(oracleDB, pathCondDB, conError);

  if (conError == L1TOMDSHelper::NO_ERROR) {
    if (m_verbose) {
      cout << "[L1TBPTX] Connected to DB with no error." << endl;
    }

    int errorRetrive;
    m_beamConfig = myOMDSHelper.getBeamConfiguration(m_lhcFill, errorRetrive);

    if (errorRetrive == L1TOMDSHelper::NO_ERROR) {
      if (m_verbose) {
        cout << "[L1TBPTX] Retriving LHC Bunch Structure: NO_ERROR" << endl;
        cout << "[L1TSync] -> LHC Bunch Structure valid=" << m_beamConfig.m_valid
             << " nBunches=" << m_beamConfig.nCollidingBunches << endl;
      }
    } else if (errorRetrive == L1TOMDSHelper::WARNING_DB_QUERY_FAILED) {
      if (m_verbose) {
        cout << "[L1TBPTX] Retriving LHC Bunch Structure: WARNING_DB_QUERY_FAILED" << endl;
      }

      int eCount = m_ErrorMonitor->getTH1()->GetBinContent(WARNING_DB_QUERY_FAILED);
      eCount++;
      m_ErrorMonitor->getTH1()->SetBinContent(WARNING_DB_QUERY_FAILED, eCount);
    } else if (errorRetrive == L1TOMDSHelper::WARNING_DB_INCORRECT_NBUNCHES) {
      if (m_verbose) {
        cout << "[L1TBPTX] Retriving LHC Bunch Structure: WARNING_DB_INCORRECT_NBUNCHES" << endl;
      }

      int eCount = m_ErrorMonitor->getTH1()->GetBinContent(WARNING_DB_INCORRECT_NBUNCHES);
      eCount++;
      m_ErrorMonitor->getTH1()->SetBinContent(WARNING_DB_INCORRECT_NBUNCHES, eCount);
    } else {
      if (m_verbose) {
        cout << "[L1TBPTX] Retriving LHC Bunch Structure: UNKNOWN" << endl;
      }
      int eCount = m_ErrorMonitor->getTH1()->GetBinContent(UNKNOWN);
      eCount++;
      m_ErrorMonitor->getTH1()->SetBinContent(UNKNOWN, eCount);
    }

  } else if (conError == L1TOMDSHelper::WARNING_DB_CONN_FAILED) {
    if (m_verbose) {
      cout << "[L1TBPTX] Connection to DB: WARNING_DB_CONN_FAILED" << endl;
    }
    int eCount = m_ErrorMonitor->getTH1()->GetBinContent(WARNING_DB_CONN_FAILED);
    eCount++;
    m_ErrorMonitor->getTH1()->SetBinContent(WARNING_DB_CONN_FAILED, eCount);
  } else {
    if (m_verbose) {
      cout << "[L1TBPTX] Connection to DB: UNKNOWN" << endl;
    }
    int eCount = m_ErrorMonitor->getTH1()->GetBinContent(UNKNOWN);
    eCount++;
    m_ErrorMonitor->getTH1()->SetBinContent(UNKNOWN, eCount);
  }
}

//_____________________________________________________________________
// Method: doFractionInSync
// Description: Produce plot with the fraction of in sync trigger for
//              LS blocks with enough statistics.
// Variable: iForce - Forces closing of all blocks and calculation of
//                    the respective fractions
// Variable: iBad   - (Only works with iForce=true) Forces the current
//                    all current blocks to be marked as bad
//_____________________________________________________________________
void L1TBPTX::doFractionInSync(bool iForce, bool iBad) {}

//_____________________________________________________________________
// Method: certifyLSBlock
// Description: Fill the trigger certification plot by blocks
// Variable: iTrigger - Which trigger to certify
// Variable: iInitLs  - Blocks initial LS
// Variable: iEndLs   - Blocks end LS
// Variable: iValue   - Value to be used to fill
//_____________________________________________________________________
void L1TBPTX::certifyLSBlock(string iTrigger, int iInitLs, int iEndLs, float iValue) {}
