// L1TGlobalUtil:  Utility class for parsing the L1 Trigger Menu

#ifndef L1TGlobal_L1TGlobalUtil_h
#define L1TGlobal_L1TGlobalUtil_h

// system include files
#include <memory>

#include <vector>

#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"

#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosFractRcd.h"
#include "CondFormats/L1TObjects/interface/L1TGlobalPrescalesVetosFract.h"

// Objects to produce for the output record.
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include "CondFormats/L1TObjects/interface/L1TUtmAlgorithm.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtilHelper.h"

#include "L1Trigger/L1TGlobal/interface/PrescalesVetosFractHelper.h"

// forward declarations

// class declaration

namespace l1t {

  // Use this to tell the EventSetup whether it should prefetch
  // data when processing beginRun or an Event or both. (This
  // depends on when retrieveL1 and retrieveL1Setup are called)
  enum class UseEventSetupIn { Run, Event, RunAndEvent };

  class L1TGlobalUtil {
  public:
    // Using this constructor will require InputTags to be specified in the configuration
    L1TGlobalUtil(edm::ParameterSet const& pset,
                  edm::ConsumesCollector&& iC,
                  UseEventSetupIn use = UseEventSetupIn::Run);

    L1TGlobalUtil(edm::ParameterSet const& pset,
                  edm::ConsumesCollector& iC,
                  UseEventSetupIn use = UseEventSetupIn::Run);

    // Using this constructor will cause it to look for valid InputTags in
    // the following ways in the specified order until they are found.
    //   1. The configuration
    //   2. Search all products from the preferred input tags for the required type
    //   3. Search all products from any other process for the required type
    template <typename T>
    L1TGlobalUtil(edm::ParameterSet const& pset,
                  edm::ConsumesCollector&& iC,
                  T& module,
                  UseEventSetupIn use = UseEventSetupIn::Run);

    template <typename T>
    L1TGlobalUtil(edm::ParameterSet const& pset,
                  edm::ConsumesCollector& iC,
                  T& module,
                  UseEventSetupIn use = UseEventSetupIn::Run);

    // Using this constructor will cause it to look for valid InputTags in
    // the following ways in the specified order until they are found.
    //   1. The constructor arguments
    //   2. The configuration
    //   3. Search all products from the preferred input tags for the required type
    //   4. Search all products from any other process for the required type
    template <typename T>
    L1TGlobalUtil(edm::ParameterSet const& pset,
                  edm::ConsumesCollector&& iC,
                  T& module,
                  edm::InputTag const& l1tAlgBlkInputTag,
                  edm::InputTag const& l1tExtBlkInputTag,
                  UseEventSetupIn use = UseEventSetupIn::Run);

    template <typename T>
    L1TGlobalUtil(edm::ParameterSet const& pset,
                  edm::ConsumesCollector& iC,
                  T& module,
                  edm::InputTag const& l1tAlgBlkInputTag,
                  edm::InputTag const& l1tExtBlkInputTag,
                  UseEventSetupIn use = UseEventSetupIn::Run);

    /// destructor
    virtual ~L1TGlobalUtil();

    /// check that the L1TGlobalUtil has been properly initialised
    bool valid() const;

    static void fillDescription(edm::ParameterSetDescription& desc,
                                edm::InputTag const& iAlg,
                                edm::InputTag const& iExt,
                                bool readPrescalesFromFile) {
      L1TGlobalUtilHelper::fillDescription(desc, iAlg, iExt, readPrescalesFromFile);
    }

    // OverridePrescalesAndMasks
    // The ability to override the prescale/mask file will not be part of the permanent interface of this class.
    // It is provided only until prescales and masks are available as CondFormats...
    // Most users should simply ignore this method and use the default ctor only!
    // Will look for prescale csv file in L1Trigger/L1TGlobal/data/Luminosity/startup/<filename>
    void OverridePrescalesAndMasks(std::string filename, unsigned int psColumn = 1.);

    /// initialize the class (mainly reserve)
    void retrieveL1(const edm::Event& iEvent, const edm::EventSetup& evSetup);  // using helper
    void retrieveL1(const edm::Event& iEvent, const edm::EventSetup& evSetup, edm::EDGetToken gtAlgToken);
    void retrieveL1Setup(const edm::EventSetup& evSetup);  // Use this one only during beginRun
    void retrieveL1Event(const edm::Event& iEvent, const edm::EventSetup& evSetup);  // using helper
    void retrieveL1Event(const edm::Event& iEvent, const edm::EventSetup& evSetup, edm::EDGetToken gtAlgToken);

    inline void setVerbosity(const int verbosity) { m_verbosity = verbosity; }

    inline bool getFinalOR() const { return m_finalOR; }

    // get the trigger bit from the name
    const bool getAlgBitFromName(const std::string& AlgName, int& bit) const;

    // get the name from the trigger bit
    const bool getAlgNameFromBit(int& bit, std::string& AlgName) const;

    // Access results for particular trigger bit
    const bool getInitialDecisionByBit(int& bit, bool& decision) const;
    const bool getIntermDecisionByBit(int& bit, bool& decision) const;
    const bool getFinalDecisionByBit(int& bit, bool& decision) const;

    // Access Prescale
    const bool getPrescaleByBit(int& bit, double& prescale) const;

    // Access Masks:
    // follows logic of uGT board:
    //       finalDecision[AlgBit]
    //           Final word is after application of prescales.
    //           A prescale = 0 effectively masks out the algorithm in the final decision word
    //
    const bool getMaskByBit(int& bit, std::vector<int>& mask) const;

    // Access results for particular trigger name
    const bool getInitialDecisionByName(const std::string& algName, bool& decision) const;
    const bool getIntermDecisionByName(const std::string& algName, bool& decision) const;
    const bool getFinalDecisionByName(const std::string& algName, bool& decision) const;

    // Access Prescales
    const bool getPrescaleByName(const std::string& algName, double& prescale) const;

    // Access Masks (see note) above
    const bool getMaskByName(const std::string& algName, std::vector<int>& mask) const;

    // Some inline commands to return the full vectors
    inline const std::vector<std::pair<std::string, bool>>& decisionsInitial() { return m_decisionsInitial; }
    inline const std::vector<std::pair<std::string, bool>>& decisionsInterm() { return m_decisionsInterm; }
    inline const std::vector<std::pair<std::string, bool>>& decisionsFinal() { return m_decisionsFinal; }

    // Access all prescales
    inline const std::vector<std::pair<std::string, double>>& prescales() { return m_prescales; }

    // Access Masks (see note) above
    inline const std::vector<std::pair<std::string, std::vector<int>>>& masks() { return m_masks; }

    // Menu names
    inline const std::string& gtTriggerMenuName() const { return m_l1GtMenu->getName(); }
    inline const std::string& gtTriggerMenuVersion() const { return m_l1GtMenu->getVersion(); }
    inline const std::string& gtTriggerMenuComment() const { return m_l1GtMenu->getComment(); }

    // Prescale Column
    inline unsigned int prescaleColumn() const { return m_PreScaleColumn; }
    inline unsigned int numberOfPreScaleColumns() const { return m_numberOfPreScaleColumns; }

  private:
    L1TGlobalUtil();

    void retrieveL1Setup(const edm::EventSetup& evSetup, bool isRun);
    void eventSetupConsumes(edm::ConsumesCollector& iC, UseEventSetupIn useEventSetupIn);

    /// clear decision vectors on a menu change
    void resetDecisionVectors();
    void resetPrescaleVectors();
    void resetMaskVectors();
    void loadPrescalesAndMasks();

    // trigger menu
    const L1TUtmTriggerMenu* m_l1GtMenu;
    unsigned long long m_l1GtMenuCacheID;

    // prescale factors
    bool m_readPrescalesFromFile;
    const l1t::PrescalesVetosFractHelper* m_l1GtPrescalesVetoes;
    unsigned long long m_l1GtPfAlgoCacheID;

    // prescales and masks
    bool m_filledPrescales;

    // algorithm maps
    //const AlgorithmMap* m_algorithmMap;
    const std::map<std::string, L1TUtmAlgorithm>* m_algorithmMap;

    // Number of physics triggers
    unsigned int m_numberPhysTriggers;
    const unsigned int m_maxNumberPhysTriggers = 512;

    //file  and container for prescale factors
    std::string m_preScaleFileName;
    unsigned int m_PreScaleColumn;
    unsigned int m_numberOfPreScaleColumns;

    std::vector<std::vector<double>> m_initialPrescaleFactorsAlgoTrig;
    const std::vector<std::vector<double>>* m_prescaleFactorsAlgoTrig;
    const std::map<int, std::vector<int>> m_initialTriggerMaskAlgoTrig;
    const std::map<int, std::vector<int>>* m_triggerMaskAlgoTrig;  // vector stores the BX

    // access to the results block from uGT
    edm::Handle<BXVector<GlobalAlgBlk>> m_uGtAlgBlk;

    // final OR
    bool m_finalOR;

    // Vectors containing the trigger name and information about that trigger
    std::vector<std::pair<std::string, bool>> m_decisionsInitial;
    std::vector<std::pair<std::string, bool>> m_decisionsInterm;
    std::vector<std::pair<std::string, bool>> m_decisionsFinal;
    std::vector<std::pair<std::string, double>> m_prescales;
    std::vector<std::pair<std::string, std::vector<int>>> m_masks;  // vector stores the bx's that are mask for given algo

    /// verbosity level
    int m_verbosity;

    std::unique_ptr<L1TGlobalUtilHelper> m_l1tGlobalUtilHelper;

    edm::ESGetToken<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd> m_L1TUtmTriggerMenuRunToken;
    edm::ESGetToken<L1TGlobalPrescalesVetosFract, L1TGlobalPrescalesVetosFractRcd>
        m_L1TGlobalPrescalesVetosFractRunToken;

    edm::ESGetToken<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd> m_L1TUtmTriggerMenuEventToken;
    edm::ESGetToken<L1TGlobalPrescalesVetosFract, L1TGlobalPrescalesVetosFractRcd>
        m_L1TGlobalPrescalesVetosFractEventToken;
  };

  template <typename T>
  L1TGlobalUtil::L1TGlobalUtil(edm::ParameterSet const& pset,
                               edm::ConsumesCollector&& iC,
                               T& module,
                               UseEventSetupIn useEventSetupIn)
      : L1TGlobalUtil(pset, iC, module, useEventSetupIn) {}

  template <typename T>
  L1TGlobalUtil::L1TGlobalUtil(edm::ParameterSet const& pset,
                               edm::ConsumesCollector& iC,
                               T& module,
                               UseEventSetupIn useEventSetupIn)
      : L1TGlobalUtil() {
    m_l1tGlobalUtilHelper = std::make_unique<L1TGlobalUtilHelper>(pset, iC, module);
    m_readPrescalesFromFile = m_l1tGlobalUtilHelper->readPrescalesFromFile();
    eventSetupConsumes(iC, useEventSetupIn);
  }

  template <typename T>
  L1TGlobalUtil::L1TGlobalUtil(edm::ParameterSet const& pset,
                               edm::ConsumesCollector&& iC,
                               T& module,
                               edm::InputTag const& l1tAlgBlkInputTag,
                               edm::InputTag const& l1tExtBlkInputTag,
                               UseEventSetupIn useEventSetupIn)
      : L1TGlobalUtil(pset, iC, module, l1tAlgBlkInputTag, l1tExtBlkInputTag, useEventSetupIn) {}

  template <typename T>
  L1TGlobalUtil::L1TGlobalUtil(edm::ParameterSet const& pset,
                               edm::ConsumesCollector& iC,
                               T& module,
                               edm::InputTag const& l1tAlgBlkInputTag,
                               edm::InputTag const& l1tExtBlkInputTag,
                               UseEventSetupIn useEventSetupIn)
      : L1TGlobalUtil() {
    m_l1tGlobalUtilHelper =
        std::make_unique<L1TGlobalUtilHelper>(pset, iC, module, l1tAlgBlkInputTag, l1tExtBlkInputTag);
    m_readPrescalesFromFile = m_l1tGlobalUtilHelper->readPrescalesFromFile();
    eventSetupConsumes(iC, useEventSetupIn);
  }
}  // namespace l1t
#endif
