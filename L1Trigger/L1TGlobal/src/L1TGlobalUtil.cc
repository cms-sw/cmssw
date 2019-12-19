// L1TGlobalUtil
//
// author: Brian Winer Ohio State
//

#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"

#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Transition.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructor
l1t::L1TGlobalUtil::L1TGlobalUtil() : m_l1GtMenu(nullptr) {
  // initialize cached IDs
  m_l1GtMenuCacheID = 0ULL;
  m_l1GtPfAlgoCacheID = 0ULL;
  m_filledPrescales = false;

  edm::FileInPath f1("L1Trigger/L1TGlobal/data/Luminosity/startup/prescale_L1TGlobal.csv");
  m_preScaleFileName = f1.fullPath();
  m_numberPhysTriggers = 512;  //need to get this out of the EventSetup
  m_PreScaleColumn = 0;
  m_readPrescalesFromFile = false;

  m_prescaleFactorsAlgoTrig = nullptr;
  m_triggerMaskAlgoTrig = nullptr;
}

l1t::L1TGlobalUtil::L1TGlobalUtil(edm::ParameterSet const& pset,
                                  edm::ConsumesCollector&& iC,
                                  UseEventSetupIn useEventSetupIn)
    : L1TGlobalUtil(pset, iC, useEventSetupIn) {}

l1t::L1TGlobalUtil::L1TGlobalUtil(edm::ParameterSet const& pset,
                                  edm::ConsumesCollector& iC,
                                  UseEventSetupIn useEventSetupIn)
    : L1TGlobalUtil() {
  m_l1tGlobalUtilHelper.reset(new L1TGlobalUtilHelper(pset, iC));
  m_readPrescalesFromFile = m_l1tGlobalUtilHelper->readPrescalesFromFile();
  eventSetupConsumes(iC, useEventSetupIn);
}

// destructor
l1t::L1TGlobalUtil::~L1TGlobalUtil() {
  // empty
}

/// check that the L1TGlobalUtil has been properly initialised
bool l1t::L1TGlobalUtil::valid() const { return m_l1GtMenuCacheID != 0ULL and m_l1GtMenu != nullptr; }

void l1t::L1TGlobalUtil::OverridePrescalesAndMasks(std::string filename, unsigned int psColumn) {
  edm::FileInPath f1("L1Trigger/L1TGlobal/data/Luminosity/startup/" + filename);
  m_preScaleFileName = f1.fullPath();
  m_PreScaleColumn = psColumn;
}

void l1t::L1TGlobalUtil::retrieveL1(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  // typically, the L1T menu and prescale table (may change only between Runs)
  bool isRun = false;
  retrieveL1Setup(evSetup, isRun);
  // typically the prescale set index used and the event by event accept/reject info (changes between Events)
  retrieveL1Event(iEvent, evSetup);
}

void l1t::L1TGlobalUtil::retrieveL1(const edm::Event& iEvent,
                                    const edm::EventSetup& evSetup,
                                    edm::EDGetToken gtAlgToken) {
  // typically, the L1T menu and prescale table (may change only between Runs)
  bool isRun = false;
  retrieveL1Setup(evSetup, isRun);
  // typically the prescale set index used and the event by event accept/reject info (changes between Events)
  retrieveL1Event(iEvent, evSetup, gtAlgToken);
}

void l1t::L1TGlobalUtil::retrieveL1Setup(const edm::EventSetup& evSetup) {
  bool isRun = true;
  retrieveL1Setup(evSetup, isRun);
}

void l1t::L1TGlobalUtil::retrieveL1Setup(const edm::EventSetup& evSetup, bool isRun) {
  // get / update the trigger menu from the EventSetup
  // local cache & check on cacheIdentifier
  auto menuRcd = evSetup.get<L1TUtmTriggerMenuRcd>();
  unsigned long long l1GtMenuCacheID = menuRcd.cacheIdentifier();

  if (m_l1GtMenuCacheID != l1GtMenuCacheID) {
    if (isRun) {
      m_l1GtMenu = &menuRcd.get(m_L1TUtmTriggerMenuRunToken);
    } else {
      m_l1GtMenu = &menuRcd.get(m_L1TUtmTriggerMenuEventToken);
    }

    //std::cout << "Attempting to fill the map " << std::endl;
    m_algorithmMap = &(m_l1GtMenu->getAlgorithmMap());

    //reset vectors since we have new menu
    resetDecisionVectors();

    m_l1GtMenuCacheID = l1GtMenuCacheID;
  }

  if (!m_readPrescalesFromFile) {
    auto vetosRcd = evSetup.get<L1TGlobalPrescalesVetosRcd>();
    unsigned long long l1GtPfAlgoCacheID = vetosRcd.cacheIdentifier();

    if (m_l1GtPfAlgoCacheID != l1GtPfAlgoCacheID) {
      //std::cout << "Reading Prescales and Masks from dB" << std::endl;

      // clear and dimension
      resetPrescaleVectors();
      resetMaskVectors();
      m_PreScaleColumn = 0;
      m_numberOfPreScaleColumns = 0;
      m_numberPhysTriggers = 0;

      const L1TGlobalPrescalesVetos* es = nullptr;
      if (isRun) {
        es = &vetosRcd.get(m_L1TGlobalPrescalesVetosRunToken);
      } else {
        es = &vetosRcd.get(m_L1TGlobalPrescalesVetosEventToken);
      }
      m_l1GtPrescalesVetoes = PrescalesVetosHelper::readFromEventSetup(es);

      m_prescaleFactorsAlgoTrig = &(m_l1GtPrescalesVetoes->prescaleTable());
      m_numberOfPreScaleColumns = m_prescaleFactorsAlgoTrig->size();
      m_numberPhysTriggers =
          (*m_prescaleFactorsAlgoTrig)[0].size();  // assumes all prescale columns are the same length

      m_triggerMaskAlgoTrig = &(m_l1GtPrescalesVetoes->triggerAlgoBxMask());

      m_l1GtPfAlgoCacheID = l1GtPfAlgoCacheID;
    }
  } else {
    //Load the prescales from external file
    // clear and dimension
    if (!m_filledPrescales) {
      resetPrescaleVectors();
      resetMaskVectors();

      loadPrescalesAndMasks();

      // Set Prescale factors to initial
      m_prescaleFactorsAlgoTrig = &m_initialPrescaleFactorsAlgoTrig;
      m_triggerMaskAlgoTrig = &m_initialTriggerMaskAlgoTrig;
      m_filledPrescales = true;
    }
  }

  //Protect against poor prescale column choice (I don't think there is a way this happen as currently structured)
  if (m_PreScaleColumn > m_prescaleFactorsAlgoTrig->size()) {
    LogTrace("l1t|Global") << "\nNo Prescale Set: " << m_PreScaleColumn
                           << "\nMax Prescale Set value : " << m_prescaleFactorsAlgoTrig->size()
                           << "\nSetting prescale column to 0" << std::endl;
    m_PreScaleColumn = 0;
  }
  //std::cout << "Using prescale column: " << m_PreScaleColumn << std::endl;
  const std::vector<int>& prescaleSet = (*m_prescaleFactorsAlgoTrig)[m_PreScaleColumn];

  for (std::map<std::string, L1TUtmAlgorithm>::const_iterator itAlgo = m_algorithmMap->begin();
       itAlgo != m_algorithmMap->end();
       itAlgo++) {
    // Get the algorithm name
    std::string algName = itAlgo->first;
    int algBit = (itAlgo->second).getIndex();  //algoBitNumber();

    (m_prescales[algBit]).first = algName;
    if (size_t(algBit) < prescaleSet.size()) {
      (m_prescales[algBit]).second = prescaleSet[algBit];
    }
    LogDebug("l1t|Global") << "Number of bunch crossings stored: " << (*m_triggerMaskAlgoTrig).size() << endl;

    const std::map<int, std::vector<int> >* triggerAlgoMaskAlgoTrig = m_triggerMaskAlgoTrig;
    std::map<int, std::vector<int> >::const_iterator it = triggerAlgoMaskAlgoTrig->begin();

    std::vector<int> maskedBxs;
    (m_masks[algBit]).first = algName;
    (m_masks[algBit]).second = maskedBxs;
    while (it != triggerAlgoMaskAlgoTrig->end()) {
      std::vector<int> masks = it->second;
      //std::cout<< "BX: " << it->first<<" VecSize: "<< masks.size();
      //std::cout << "\tMasked algos: ";
      for (unsigned int imask = 0; imask < masks.size(); imask++) {
        if (masks.at(imask) == algBit)
          maskedBxs.push_back(it->first);
        // std::cout << "\t" << masks.at(imask);
      }
      //std::cout << "\n";
      it++;
    }

    if (!maskedBxs.empty()) {
      LogDebug("l1t|Global") << "i Algo: " << algBit << "\t" << algName << " masked\n";
      for (unsigned int ibx = 0; ibx < maskedBxs.size(); ibx++) {
        // std::cout << "\t" << maskedBxs.at(ibx);
        (m_masks[algBit]).second = maskedBxs;
      }
    }
  }
}

void l1t::L1TGlobalUtil::retrieveL1Event(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  retrieveL1Event(iEvent, evSetup, m_l1tGlobalUtilHelper->l1tAlgBlkToken());
}

void l1t::L1TGlobalUtil::retrieveL1Event(const edm::Event& iEvent,
                                         const edm::EventSetup& evSetup,
                                         edm::EDGetToken gtAlgToken) {
  // Get the Global Trigger Output Algorithm block
  iEvent.getByToken(gtAlgToken, m_uGtAlgBlk);
  m_finalOR = false;

  //Make sure we have a valid AlgBlk
  if (m_uGtAlgBlk.isValid()) {
    // get the GlabalAlgBlk (Stupid find better way) of BX=0
    std::vector<GlobalAlgBlk>::const_iterator algBlk = m_uGtAlgBlk->begin(0);
    if (algBlk != m_uGtAlgBlk->end(0)) {
      if (!m_readPrescalesFromFile) {
        m_PreScaleColumn = static_cast<unsigned int>(algBlk->getPreScColumn());

        // Fix for MC prescale column being set to index+1 in early versions of uGT emulator
        if (iEvent.run() == 1) {
          if (m_prescaleFactorsAlgoTrig->size() == 1 && m_PreScaleColumn == 1)
            m_PreScaleColumn = 0;
        }

        // add protection against out-of-bound index for prescale column
        if (m_PreScaleColumn >= m_prescaleFactorsAlgoTrig->size()) {
          LogDebug("l1t|Global") << "Prescale column extracted from GlobalAlgBlk too large: " << m_PreScaleColumn
                                 << "\tMaximum value allowed: " << m_prescaleFactorsAlgoTrig->size() - 1
                                 << "\tResetting prescale column to 0" << std::endl;
          m_PreScaleColumn = 0;
        }
      }
      const std::vector<int>& prescaleSet = (*m_prescaleFactorsAlgoTrig)[m_PreScaleColumn];

      // Grab the final OR from the AlgBlk,
      m_finalOR = algBlk->getFinalOR();

      // Make a map of the trigger name and whether it passed various stages (initial,prescale,final)
      // Note: might be able to improve performance by not full remaking map with names each time
      for (std::map<std::string, L1TUtmAlgorithm>::const_iterator itAlgo = m_algorithmMap->begin();
           itAlgo != m_algorithmMap->end();
           itAlgo++) {
        // Get the algorithm name
        std::string algName = itAlgo->first;
        int algBit = (itAlgo->second).getIndex();  //algoBitNumber();

        bool decisionInitial = algBlk->getAlgoDecisionInitial(algBit);
        (m_decisionsInitial[algBit]).first = algName;
        (m_decisionsInitial[algBit]).second = decisionInitial;

        bool decisionInterm = algBlk->getAlgoDecisionInterm(algBit);
        (m_decisionsInterm[algBit]).first = algName;
        (m_decisionsInterm[algBit]).second = decisionInterm;

        bool decisionFinal = algBlk->getAlgoDecisionFinal(algBit);
        (m_decisionsFinal[algBit]).first = algName;
        (m_decisionsFinal[algBit]).second = decisionFinal;

        (m_prescales[algBit]).first = algName;
        if (size_t(algBit) < prescaleSet.size()) {
          (m_prescales[algBit]).second = prescaleSet[algBit];
        }
      }
    } else {
      //cout << "Error empty AlgBlk recovered.\n";
    }
  } else {
    //cout<< "Error no valid uGT Algorithm Data with Token provided " << endl;
  }
}

void l1t::L1TGlobalUtil::loadPrescalesAndMasks() {
  std::ifstream inputPrescaleFile;
  //std::cout << "Reading prescales from file: " << m_preScaleFileName << std::endl;
  inputPrescaleFile.open(m_preScaleFileName);

  std::vector<std::vector<int> > vec;
  std::vector<std::vector<int> > prescale_vec;

  if (inputPrescaleFile) {
    std::string prefix1("#");
    std::string prefix2("-1");

    std::string line;

    bool first = true;

    while (getline(inputPrescaleFile, line)) {
      if (!line.compare(0, prefix1.size(), prefix1))
        continue;
      //if( !line.compare(0, prefix2.size(), prefix2) ) continue;

      istringstream split(line);
      int value;
      int col = 0;
      char sep;

      while (split >> value) {
        if (first) {
          // Each new value read on line 1 should create a new inner vector
          vec.push_back(std::vector<int>());
        }

        vec[col].push_back(value);
        ++col;

        // read past the separator
        split >> sep;
      }

      // Finished reading line 1 and creating as many inner
      // vectors as required
      first = false;
    }

    int NumPrescaleSets = 0;
    for (int iCol = 0; iCol < int(vec.size()); iCol++) {
      if (!vec[iCol].empty()) {
        int firstRow = vec[iCol][0];

        if (firstRow >= 0)
          NumPrescaleSets++;
        //else if( firstRow==-2 ) maskColumn = iCol;
        //else if( firstRow==-3 ) maskVetoColumn = iCol;
      }
    }

    //std::cout << "NumPrescaleSets= " << NumPrescaleSets << std::endl;
    if (NumPrescaleSets > 0) {
      // Fill default prescale set
      for (int iSet = 0; iSet < NumPrescaleSets; iSet++) {
        prescale_vec.push_back(std::vector<int>());
        for (unsigned int iBit = 0; iBit < m_numberPhysTriggers; ++iBit) {
          int inputDefaultPrescale = 1;
          prescale_vec[iSet].push_back(inputDefaultPrescale);
        }
      }

      // Fill non-trivial prescale set
      for (int iBit = 1; iBit < int(vec[0].size()); iBit++) {
        unsigned int algoBit = vec[0][iBit];
        // algoBit must be less than the number of triggers
        if (algoBit < m_numberPhysTriggers) {
          for (int iSet = 0; iSet < int(vec.size()); iSet++) {
            int useSet = -1;
            if (!vec[iSet].empty()) {
              useSet = vec[iSet][0];
            }
            useSet -= 1;

            if (useSet < 0)
              continue;

            int prescale = vec[iSet][iBit];
            prescale_vec[useSet][algoBit] = prescale;
          }
        } else {
          LogTrace("l1t|Global") << "\nPrescale file has algo bit: " << algoBit
                                 << "\nThis is larger than the number of triggers: " << m_numberPhysTriggers
                                 << "\nSomething is wrong. Ignoring." << std::endl;
        }
      }
    }

  } else {
    LogTrace("l1t|Global") << "\nCould not find file: " << m_preScaleFileName
                           << "\nFilling the prescale vectors with prescale 1"
                           << "\nSetting prescale set to 0" << std::endl;

    m_PreScaleColumn = 0;

    for (int col = 0; col < 1; col++) {
      prescale_vec.push_back(std::vector<int>());
      for (unsigned int iBit = 0; iBit < m_numberPhysTriggers; ++iBit) {
        int inputDefaultPrescale = 0;
        prescale_vec[col].push_back(inputDefaultPrescale);
      }
    }
  }

  inputPrescaleFile.close();

  m_initialPrescaleFactorsAlgoTrig = prescale_vec;
  // setting of bx masks from an input file not enabled; do not see a use case at the moment
  std::map<int, std::vector<int> > m_initialTriggerMaskAlgoTrig;
}

void l1t::L1TGlobalUtil::eventSetupConsumes(edm::ConsumesCollector& iC, UseEventSetupIn useEventSetupIn) {
  if (useEventSetupIn == UseEventSetupIn::Run || useEventSetupIn == UseEventSetupIn::RunAndEvent) {
    m_L1TUtmTriggerMenuRunToken = iC.esConsumes<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd, edm::Transition::BeginRun>();
    if (!m_readPrescalesFromFile) {
      m_L1TGlobalPrescalesVetosRunToken =
          iC.esConsumes<L1TGlobalPrescalesVetos, L1TGlobalPrescalesVetosRcd, edm::Transition::BeginRun>();
    }
  }
  if (useEventSetupIn == UseEventSetupIn::Event || useEventSetupIn == UseEventSetupIn::RunAndEvent) {
    m_L1TUtmTriggerMenuEventToken = iC.esConsumes<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd>();
    if (!m_readPrescalesFromFile) {
      m_L1TGlobalPrescalesVetosEventToken = iC.esConsumes<L1TGlobalPrescalesVetos, L1TGlobalPrescalesVetosRcd>();
    }
  }
}

void l1t::L1TGlobalUtil::resetDecisionVectors() {
  // Reset all the vector contents with null information
  m_decisionsInitial.clear();
  m_decisionsInitial.resize(m_maxNumberPhysTriggers);
  m_decisionsInterm.clear();
  m_decisionsInterm.resize(m_maxNumberPhysTriggers);
  m_decisionsFinal.clear();
  m_decisionsFinal.resize(m_maxNumberPhysTriggers);

  for (unsigned int algBit = 0; algBit < m_maxNumberPhysTriggers; algBit++) {
    (m_decisionsInitial[algBit]).first = "NULL";
    (m_decisionsInitial[algBit]).second = false;

    (m_decisionsInterm[algBit]).first = "NULL";
    (m_decisionsInterm[algBit]).second = false;

    (m_decisionsFinal[algBit]).first = "NULL";
    (m_decisionsFinal[algBit]).second = false;
  }
}

void l1t::L1TGlobalUtil::resetPrescaleVectors() {
  // Reset all the vector contents with null information
  m_prescales.clear();
  m_prescales.resize(m_maxNumberPhysTriggers);

  for (unsigned int algBit = 0; algBit < m_maxNumberPhysTriggers; algBit++) {
    (m_prescales[algBit]).first = "NULL";
    (m_prescales[algBit]).second = 1;
  }
}

void l1t::L1TGlobalUtil::resetMaskVectors() {
  // Reset all the vector contents with null information
  m_masks.clear();
  m_masks.resize(m_maxNumberPhysTriggers);

  for (unsigned int algBit = 0; algBit < m_maxNumberPhysTriggers; algBit++) {
    (m_masks[algBit]).first = "NULL";
    // ccla (m_masks[algBit]).second = true;
  }
}

const bool l1t::L1TGlobalUtil::getAlgBitFromName(const std::string& algName, int& bit) const {
  std::map<std::string, L1TUtmAlgorithm>::const_iterator itAlgo = m_algorithmMap->find(algName);
  if (itAlgo != m_algorithmMap->end()) {
    bit = (itAlgo->second).getIndex();  //algoBitNumber();
    return true;
  }

  return false;  //did not find anything by that name
}

const bool l1t::L1TGlobalUtil::getAlgNameFromBit(int& bit, std::string& algName) const {
  // since we are just looking up the name, doesn't matter which vector we get it from
  if ((m_decisionsInitial[bit]).first != "NULL") {
    algName = (m_decisionsInitial[bit]).first;
    return true;
  }
  return false;  //No name associated with this bit
}

const bool l1t::L1TGlobalUtil::getInitialDecisionByBit(int& bit, bool& decision) const {
  /*
    for(std::vector<GlobalAlgBlk>::const_iterator algBlk = m_uGtAlgBlk->begin(0); algBlk != m_uGtAlgBlk->end(0); ++algBlk) {
      decision = algBlk->getAlgoDecisionFinal(bit);
    }
  */
  // Need some check that this is a valid bit
  if ((m_decisionsInitial[bit]).first != "NULL") {
    decision = (m_decisionsInitial[bit]).second;
    return true;
  }

  return false;  //couldn't get the information requested.
}
const bool l1t::L1TGlobalUtil::getIntermDecisionByBit(int& bit, bool& decision) const {
  // Need some check that this is a valid bit
  if ((m_decisionsInterm[bit]).first != "NULL") {
    decision = (m_decisionsInterm[bit]).second;
    return true;
  }

  return false;  //couldn't get the information requested.
}
const bool l1t::L1TGlobalUtil::getFinalDecisionByBit(int& bit, bool& decision) const {
  // Need some check that this is a valid bit
  if ((m_decisionsFinal[bit]).first != "NULL") {
    decision = (m_decisionsFinal[bit]).second;
    return true;
  }

  return false;  //couldn't get the information requested.
}
const bool l1t::L1TGlobalUtil::getPrescaleByBit(int& bit, int& prescale) const {
  // Need some check that this is a valid bit
  if ((m_prescales[bit]).first != "NULL") {
    prescale = (m_prescales[bit]).second;
    return true;
  }

  return false;  //couldn't get the information requested.
}
const bool l1t::L1TGlobalUtil::getMaskByBit(int& bit, std::vector<int>& mask) const {
  // Need some check that this is a valid bit
  if ((m_masks[bit]).first != "NULL") {
    mask = (m_masks[bit]).second;
    return true;
  }

  return false;  //couldn't get the information requested.
}

const bool l1t::L1TGlobalUtil::getInitialDecisionByName(const std::string& algName, bool& decision) const {
  int bit = -1;
  if (getAlgBitFromName(algName, bit)) {
    decision = (m_decisionsInitial[bit]).second;
    return true;
  }

  return false;  //trigger name was not the menu.
}

const bool l1t::L1TGlobalUtil::getIntermDecisionByName(const std::string& algName, bool& decision) const {
  int bit = -1;
  if (getAlgBitFromName(algName, bit)) {
    decision = (m_decisionsInterm[bit]).second;
    return true;
  }

  return false;  //trigger name was not the menu.
}

const bool l1t::L1TGlobalUtil::getFinalDecisionByName(const std::string& algName, bool& decision) const {
  int bit = -1;
  if (getAlgBitFromName(algName, bit)) {
    decision = (m_decisionsFinal[bit]).second;
    return true;
  }

  return false;  //trigger name was not the menu.
}
const bool l1t::L1TGlobalUtil::getPrescaleByName(const std::string& algName, int& prescale) const {
  int bit = -1;
  if (getAlgBitFromName(algName, bit)) {
    prescale = (m_prescales[bit]).second;
    return true;
  }

  return false;  //trigger name was not the menu.
}
const bool l1t::L1TGlobalUtil::getMaskByName(const std::string& algName, std::vector<int>& mask) const {
  int bit = -1;
  if (getAlgBitFromName(algName, bit)) {
    mask = (m_masks[bit]).second;
    return true;
  }

  return false;  //trigger name was not the menu.
}
