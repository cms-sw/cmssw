#ifndef GlobalTriggerAnalyzer_L1GtUtils_h
#define GlobalTriggerAnalyzer_L1GtUtils_h

/**
 * \class L1GtUtils
 *
 *
 * Description: various methods for L1 GT, to be called in an EDM analyzer, producer or filter.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// system include files
#include <memory>
#include <string>
#include <utility>

// user include files
#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoTechTrigRcd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtTriggerMenuLite.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtilsHelper.h"

// forward declarations
class L1GtStableParameters;
class L1GtPrescaleFactors;
class L1GtTriggerMask;
class L1GtTriggerMenu;

class L1GlobalTriggerReadoutRecord;
class L1GlobalTriggerRecord;

// class declaration

class L1GtUtils {
public:
  // Use this enum to tell the EventSetup whether it should prefetch
  // data when processing beginRun or an Event or both. (This
  // depends on when retrieveL1EventSetup is called which can
  // be a direct call or also indirectly through a call to one
  // or both versions of getL1GtRunCache. Also note that getL1GtRunCache
  // has an argument that disables EventSetup calls and if the Run
  // version of the function is called then the Event version of the
  // function does not get EventSetup data)
  enum class UseEventSetupIn { Run, Event, RunAndEvent, Nothing };

  // Using this constructor will require InputTags to be specified in the configuration
  L1GtUtils(edm::ParameterSet const& pset,
            edm::ConsumesCollector&& iC,
            bool useL1GtTriggerMenuLite,
            UseEventSetupIn use = UseEventSetupIn::Run);

  L1GtUtils(edm::ParameterSet const& pset,
            edm::ConsumesCollector& iC,
            bool useL1GtTriggerMenuLite,
            UseEventSetupIn use = UseEventSetupIn::Run);

  // Using this constructor will cause it to look for valid InputTags in
  // the following ways in the specified order until they are found.
  //   1. The configuration
  //   2. Search all products from the preferred input tags for the required type
  //   3. Search all products from any other process for the required type
  template <typename T>
  L1GtUtils(edm::ParameterSet const& pset,
            edm::ConsumesCollector&& iC,
            bool useL1GtTriggerMenuLite,
            T& module,
            UseEventSetupIn use = UseEventSetupIn::Run);

  template <typename T>
  L1GtUtils(edm::ParameterSet const& pset,
            edm::ConsumesCollector& iC,
            bool useL1GtTriggerMenuLite,
            T& module,
            UseEventSetupIn use = UseEventSetupIn::Run);

  // Using this constructor will cause it to look for valid InputTags in
  // the following ways in the specified order until they are found.
  //   1. The constructor arguments
  //   2. The configuration
  //   3. Search all products from the preferred input tags for the required type
  //   4. Search all products from any other process for the required type
  template <typename T>
  L1GtUtils(edm::ParameterSet const& pset,
            edm::ConsumesCollector&& iC,
            bool useL1GtTriggerMenuLite,
            T& module,
            edm::InputTag const& l1GtRecordInputTag,
            edm::InputTag const& l1GtReadoutRecordInputTag,
            edm::InputTag const& l1GtTriggerMenuLiteInputTag,
            UseEventSetupIn use = UseEventSetupIn::Run);

  template <typename T>
  L1GtUtils(edm::ParameterSet const& pset,
            edm::ConsumesCollector& iC,
            bool useL1GtTriggerMenuLite,
            T& module,
            edm::InputTag const& l1GtRecordInputTag,
            edm::InputTag const& l1GtReadoutRecordInputTag,
            edm::InputTag const& l1GtTriggerMenuLiteInputTag,
            UseEventSetupIn use = UseEventSetupIn::Run);

  /// destructor
  virtual ~L1GtUtils();

  static void fillDescription(edm::ParameterSetDescription& desc) { L1GtUtilsHelper::fillDescription(desc); }

public:
  enum TriggerCategory { AlgorithmTrigger = 0, TechnicalTrigger = 1 };

  /**
     * \class L1GtUtils
     *
     *
     * Description: return L1 trigger results for a logical expression.
     *
     * Implementation:
     *    Return L1 trigger results for a logical expression of algorithm or technical triggers.
     *    Mixture of algorithm and technical triggers in the logical expression is allowed only if
     *    trigger names or aliases are used. Mixing bit numbers and names or aliases is not supported.
     *    If the expression has bit numbers, they are assumed to be technical triggers.
     *
     * \author: Vasile Mihai Ghete - HEPHY Vienna
     *      *
     */
  class LogicalExpressionL1Results {
  public:
    /// constructor(s)

    /// trigger decisions, prescale factors and masks from GT record(s)
    explicit LogicalExpressionL1Results(const std::string&, L1GtUtils&);

    /// destructor
    ~LogicalExpressionL1Results();

  public:
    /// return true if the logical expression is syntactically correct
    inline bool isValid() { return m_validLogicalExpression; }

    /// update quantities related to the logical expression at the beginning of the run
    ///     new logical expression, replacing the logical expression given the in previous run
    const int logicalExpressionRunUpdate(const edm::Run&, const edm::EventSetup&, const std::string&);
    //
    ///     keep the logical expression given in the previous run
    const int logicalExpressionRunUpdate(const edm::Run&, const edm::EventSetup&);

    /// list of triggers in the logical expression, trigger decisions, prescale factors and masks, error codes

    inline const std::vector<L1GtLogicParser::OperandToken>& expL1Triggers() { return m_expL1Triggers; }
    const std::vector<std::pair<std::string, bool> >& decisionsBeforeMask();
    const std::vector<std::pair<std::string, bool> >& decisionsAfterMask();
    const std::vector<std::pair<std::string, int> >& prescaleFactors();
    const std::vector<std::pair<std::string, int> >& triggerMasks();

    const std::vector<std::pair<std::string, int> >& errorCodes(const edm::Event&);

  private:
    /// parse the logical expression, initialize the private members to required size
    /// such that one can just reset them
    bool initialize();

    /// reset for each L1 trigger the value from pair.second
    void reset(const std::vector<std::pair<std::string, bool> >&) const;
    void reset(const std::vector<std::pair<std::string, int> >&) const;

    void l1Results(const edm::Event& iEvent);

  private:
    /// private members as input parameters

    /// logical expression
    std::string m_logicalExpression;

    L1GtUtils& m_l1GtUtils;

  private:
    // private members

    /// code for L1 trigger configuration
    int m_l1ConfCode;

    /// true if valid L1 configuration - if not, reset all quantities and return
    bool m_validL1Configuration;

    /// true if the logical expression uses accepted L1GtLogicParser operators
    bool m_validLogicalExpression;

    /// set to true if the method l1Results was called once
    bool m_l1ResultsAlreadyCalled;

    std::vector<L1GtLogicParser::OperandToken> m_expL1Triggers;
    size_t m_expL1TriggersSize;

    /// trigger category for each L1 trigger in the logical expression
    std::vector<L1GtUtils::TriggerCategory> m_expTriggerCategory;

    /// flag true, if the logical expression is built from technical trigger bits
    bool m_expBitsTechTrigger;

    /// for each L1 trigger in the logical expression, true if the trigger is found
    /// in the current L1 menu
    std::vector<bool> m_expTriggerInMenu;

    ///
    std::vector<std::pair<std::string, bool> > m_decisionsBeforeMask;
    std::vector<std::pair<std::string, bool> > m_decisionsAfterMask;
    std::vector<std::pair<std::string, int> > m_prescaleFactors;
    std::vector<std::pair<std::string, int> > m_triggerMasks;
    std::vector<std::pair<std::string, int> > m_errorCodes;
  };

public:
  /// public methods

  // enum to string for TriggerCategory
  const std::string triggerCategory(const TriggerCategory&) const;

  /// retrieve all the relevant L1 trigger event setup records and cache them to improve the speed
  void retrieveL1EventSetup(const edm::EventSetup&, bool isRun = true);

  /// retrieve L1GtTriggerMenuLite (per run product) and cache it to improve the speed

  ///    for use in beginRun(const edm::Run&, const edm::EventSetup&);
  void retrieveL1GtTriggerMenuLite(const edm::Run&);

  /// get all the run-constant quantities for L1 trigger and cache them

  ///    for use in beginRun(const edm::Run&, const edm::EventSetup&);
  void getL1GtRunCache(const edm::Run&, const edm::EventSetup&, const bool, const bool);

  ///    for use in analyze(const edm::Event&, const edm::EventSetup&)
  void getL1GtRunCache(const edm::Event&, const edm::EventSetup&, const bool, const bool);

  /// return the trigger "category" trigCategory
  ///    algorithm trigger alias or algorithm trigger name AlgorithmTrigger = 0,
  ///    technical trigger TechnicalTrigger = 1
  /// and its bit number
  ///
  /// in case the algorithm trigger / technical trigger is not in the menu,
  /// the returned function is false, the trigger category is irrelevant
  /// (default value is AlgorithmTrigger), and the value of the bit number is -1

  const bool l1AlgoTechTrigBitNumber(const std::string& nameAlgoTechTrig,
                                     TriggerCategory& trigCategory,
                                     int& bitNumber) const;

  /// return the trigger name and alias for a given trigger category and a given
  /// bit number
  ///
  /// in case no algorithm trigger / technical trigger is defined for that bit in the menu,
  /// the returned function is false, and the name and the alias is empty

  const bool l1TriggerNameFromBit(const int& bitNumber,
                                  const TriggerCategory& trigCategory,
                                  std::string& aliasL1Trigger,
                                  std::string& nameL1Trigger) const;

  /// return results for a given algorithm or technical trigger,
  /// input:
  ///   event
  ///   algorithm trigger name or alias, or technical trigger name
  /// output (by reference):
  ///    decision before mask,
  ///    decision after mask,
  ///    prescale factor
  ///    trigger mask
  /// return: integer error code

  const int l1Results(const edm::Event& iEvent,
                      const std::string& nameAlgoTechTrig,
                      bool& decisionBeforeMask,
                      bool& decisionAfterMask,
                      int& prescaleFactor,
                      int& triggerMask) const;

  /// for the functions decisionBeforeMask, decisionAfterMask, decision
  /// prescaleFactor, trigger mask:
  ///
  /// input:
  ///   event, event setup
  ///   algorithm trigger name or alias, or technical trigger name
  /// output (by reference):
  ///    error code
  /// return: the corresponding quantity

  ///   return decision before trigger mask for a given algorithm or technical trigger
  const bool decisionBeforeMask(const edm::Event& iEvent, const std::string& nameAlgoTechTrig, int& errorCode) const;

  ///   return decision after trigger mask for a given algorithm or technical trigger
  const bool decisionAfterMask(const edm::Event& iEvent, const std::string& nameAlgoTechTrig, int& errorCode) const;

  ///   return decision after trigger mask for a given algorithm or technical trigger
  ///          function identical with decisionAfterMask
  const bool decision(const edm::Event& iEvent, const std::string& nameAlgoTechTrig, int& errorCode) const;

  ///   return prescale factor for a given algorithm or technical trigger
  const int prescaleFactor(const edm::Event& iEvent, const std::string& nameAlgoTechTrig, int& errorCode) const;

  ///   return trigger mask for a given algorithm or technical trigger
  const int triggerMask(const edm::Event& iEvent, const std::string& nameAlgoTechTrig, int& errorCode) const;

  ///     faster than previous two methods - one needs in fact for the
  ///     masks the event setup only
  const int triggerMask(const std::string& nameAlgoTechTrig, int& errorCode) const;

  /// return the index of the actual set of prescale factors used for the
  /// event (must be the same for all events in the luminosity block,
  /// if no errors)
  ///

  const int prescaleFactorSetIndex(const edm::Event& iEvent, const TriggerCategory& trigCategory, int& errorCode) const;

  /// return the actual set of prescale factors used for the
  /// event (must be the same for all events in the luminosity block,
  /// if no errors)

  const std::vector<int>& prescaleFactorSet(const edm::Event& iEvent,
                                            const TriggerCategory& trigCategory,
                                            int& errorCode);

  /// return the set of trigger masks for the physics partition (partition zero)
  /// used for the event (remain the same in the whole run, if no errors)
  const std::vector<unsigned int>& triggerMaskSet(const TriggerCategory& trigCategory, int& errorCode);

  /// return the L1 trigger menu name
  const std::string& l1TriggerMenu() const;

  /// return the L1 trigger menu implementation
  const std::string& l1TriggerMenuImplementation() const;

  /// return a pointer to the L1 trigger menu from event setup
  const L1GtTriggerMenu* ptrL1TriggerMenuEventSetup(int& errorCode);

  /// return a pointer to the L1GtTriggerMenuLite product
  const L1GtTriggerMenuLite* ptrL1GtTriggerMenuLite(int& errorCode);

  /// check if L1 trigger configuration is available
  /// return false and an error code if configuration is not available
  const bool availableL1Configuration(int& errorCode, int& l1ConfCode) const;

private:
  static const std::string EmptyString;
  static const int L1GtNotValidError;

  /// return the trigger result given bit number and decision word
  /// errorCode != 0 if bit number greater than size of decision word
  /// print in debug mode a message in case of error
  const bool trigResult(const DecisionWord& decWord,
                        const int bitNumber,
                        const std::string& nameAlgoTechTrig,
                        const TriggerCategory& trigCategory,
                        int& errorCode) const;

private:
  L1GtUtils(edm::ConsumesCollector&, UseEventSetupIn);

  /// event setup cached stuff

  /// stable parameters
  const L1GtStableParameters* m_l1GtStablePar;
  unsigned long long m_l1GtStableParCacheID;

  /// number of algorithm triggers
  unsigned int m_numberAlgorithmTriggers;

  /// number of technical triggers
  unsigned int m_numberTechnicalTriggers;

  /// prescale factors
  const L1GtPrescaleFactors* m_l1GtPfAlgo;
  unsigned long long m_l1GtPfAlgoCacheID;

  const L1GtPrescaleFactors* m_l1GtPfTech;
  unsigned long long m_l1GtPfTechCacheID;

  const std::vector<std::vector<int> >* m_prescaleFactorsAlgoTrig;
  const std::vector<std::vector<int> >* m_prescaleFactorsTechTrig;

  /// trigger masks & veto masks
  const L1GtTriggerMask* m_l1GtTmAlgo;
  unsigned long long m_l1GtTmAlgoCacheID;

  const L1GtTriggerMask* m_l1GtTmTech;
  unsigned long long m_l1GtTmTechCacheID;

  const L1GtTriggerMask* m_l1GtTmVetoAlgo;
  unsigned long long m_l1GtTmVetoAlgoCacheID;

  const L1GtTriggerMask* m_l1GtTmVetoTech;
  unsigned long long m_l1GtTmVetoTechCacheID;

  const std::vector<unsigned int>* m_triggerMaskAlgoTrig;
  const std::vector<unsigned int>* m_triggerMaskTechTrig;

  const std::vector<unsigned int>* m_triggerMaskVetoAlgoTrig;
  const std::vector<unsigned int>* m_triggerMaskVetoTechTrig;

  // trigger menu
  const L1GtTriggerMenu* m_l1GtMenu;
  unsigned long long m_l1GtMenuCacheID;

  const AlgorithmMap* m_algorithmMap;
  const AlgorithmMap* m_algorithmAliasMap;
  const AlgorithmMap* m_technicalTriggerMap;

  bool m_l1EventSetupValid;

  /// L1GtTriggerMenuLite cached stuff

  /// L1GtTriggerMenuLite
  const L1GtTriggerMenuLite* m_l1GtMenuLite;

  const L1GtTriggerMenuLite::L1TriggerMap* m_algorithmMapLite;
  const L1GtTriggerMenuLite::L1TriggerMap* m_algorithmAliasMapLite;
  const L1GtTriggerMenuLite::L1TriggerMap* m_technicalTriggerMapLite;

  const std::vector<unsigned int>* m_triggerMaskAlgoTrigLite;
  const std::vector<unsigned int>* m_triggerMaskTechTrigLite;

  const std::vector<std::vector<int> >* m_prescaleFactorsAlgoTrigLite;
  const std::vector<std::vector<int> >* m_prescaleFactorsTechTrigLite;

  bool m_l1GtMenuLiteValid;

  /// flag for call of getL1GtRunCache in beginRun
  bool m_beginRunCache;

  /// run cache ID
  edm::RunID m_runIDCache;

private:
  /// index of physics DAQ partition
  unsigned int m_physicsDaqPartition;

  std::vector<unsigned int> m_triggerMaskSet;
  std::vector<int> m_prescaleFactorSet;

  /// flags to check which method was used to retrieve L1 trigger configuration
  bool m_retrieveL1EventSetup;
  bool m_retrieveL1GtTriggerMenuLite;

  std::unique_ptr<L1GtUtilsHelper> m_l1GtUtilsHelper;

  // beginRun EventSetup tokens
  edm::ESGetToken<L1GtStableParameters, L1GtStableParametersRcd> m_L1GtStableParametersRunToken;
  edm::ESGetToken<L1GtPrescaleFactors, L1GtPrescaleFactorsAlgoTrigRcd> m_L1GtPrescaleFactorsAlgoTrigRunToken;
  edm::ESGetToken<L1GtPrescaleFactors, L1GtPrescaleFactorsTechTrigRcd> m_L1GtPrescaleFactorsTechTrigRunToken;
  edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskAlgoTrigRcd> m_L1GtTriggerMaskAlgoTrigRunToken;
  edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskTechTrigRcd> m_L1GtTriggerMaskTechTrigRunToken;
  edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskVetoAlgoTrigRcd> m_L1GtTriggerMaskVetoAlgoTrigRunToken;
  edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskVetoTechTrigRcd> m_L1GtTriggerMaskVetoTechTrigRunToken;
  edm::ESGetToken<L1GtTriggerMenu, L1GtTriggerMenuRcd> m_L1GtTriggerMenuRunToken;

  // event transition EventSetup tokens (same as run tokens except a different name)
  edm::ESGetToken<L1GtStableParameters, L1GtStableParametersRcd> m_L1GtStableParametersEventToken;
  edm::ESGetToken<L1GtPrescaleFactors, L1GtPrescaleFactorsAlgoTrigRcd> m_L1GtPrescaleFactorsAlgoTrigEventToken;
  edm::ESGetToken<L1GtPrescaleFactors, L1GtPrescaleFactorsTechTrigRcd> m_L1GtPrescaleFactorsTechTrigEventToken;
  edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskAlgoTrigRcd> m_L1GtTriggerMaskAlgoTrigEventToken;
  edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskTechTrigRcd> m_L1GtTriggerMaskTechTrigEventToken;
  edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskVetoAlgoTrigRcd> m_L1GtTriggerMaskVetoAlgoTrigEventToken;
  edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskVetoTechTrigRcd> m_L1GtTriggerMaskVetoTechTrigEventToken;
  edm::ESGetToken<L1GtTriggerMenu, L1GtTriggerMenuRcd> m_L1GtTriggerMenuEventToken;
};

template <typename T>
L1GtUtils::L1GtUtils(edm::ParameterSet const& pset,
                     edm::ConsumesCollector&& iC,
                     bool useL1GtTriggerMenuLite,
                     T& module,
                     UseEventSetupIn useEventSetupIn)
    : L1GtUtils(pset, iC, useL1GtTriggerMenuLite, module, useEventSetupIn) {}

template <typename T>
L1GtUtils::L1GtUtils(edm::ParameterSet const& pset,
                     edm::ConsumesCollector& iC,
                     bool useL1GtTriggerMenuLite,
                     T& module,
                     UseEventSetupIn useEventSetupIn)
    : L1GtUtils(iC, useEventSetupIn) {
  m_l1GtUtilsHelper.reset(new L1GtUtilsHelper(pset, iC, useL1GtTriggerMenuLite, module));
}

template <typename T>
L1GtUtils::L1GtUtils(edm::ParameterSet const& pset,
                     edm::ConsumesCollector&& iC,
                     bool useL1GtTriggerMenuLite,
                     T& module,
                     edm::InputTag const& l1GtRecordInputTag,
                     edm::InputTag const& l1GtReadoutRecordInputTag,
                     edm::InputTag const& l1GtTriggerMenuLiteInputTag,
                     UseEventSetupIn useEventSetupIn)
    : L1GtUtils(pset,
                iC,
                useL1GtTriggerMenuLite,
                module,
                l1GtRecordInputTag,
                l1GtReadoutRecordInputTag,
                l1GtTriggerMenuLiteInputTag,
                useEventSetupIn) {}

template <typename T>
L1GtUtils::L1GtUtils(edm::ParameterSet const& pset,
                     edm::ConsumesCollector& iC,
                     bool useL1GtTriggerMenuLite,
                     T& module,
                     edm::InputTag const& l1GtRecordInputTag,
                     edm::InputTag const& l1GtReadoutRecordInputTag,
                     edm::InputTag const& l1GtTriggerMenuLiteInputTag,
                     UseEventSetupIn useEventSetupIn)
    : L1GtUtils(iC, useEventSetupIn) {
  m_l1GtUtilsHelper.reset(new L1GtUtilsHelper(pset,
                                              iC,
                                              useL1GtTriggerMenuLite,
                                              module,
                                              l1GtRecordInputTag,
                                              l1GtReadoutRecordInputTag,
                                              l1GtTriggerMenuLiteInputTag));
}

#endif /*GlobalTriggerAnalyzer_L1GtUtils_h*/
