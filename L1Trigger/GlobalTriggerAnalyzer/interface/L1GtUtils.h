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
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <string>
#include <utility>

// user include files

#include "DataFormats/L1GlobalTrigger/interface/L1GtTriggerMenuLite.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

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

    /// constructor
    explicit L1GtUtils();

    /// destructor
    virtual ~L1GtUtils();

public:

    enum TriggerCategory {
        AlgorithmTrigger = 0, TechnicalTrigger = 1
    };

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

        /// trigger decisions, prescale factors and masks from GT record(s) with input tag(s) 
        /// from provenance
        explicit LogicalExpressionL1Results(const std::string&, L1GtUtils&);

        /// trigger decisions, prescale factors and masks from GT record(s) with input tag(s) 
        /// explicitly given
        explicit LogicalExpressionL1Results(const std::string&, L1GtUtils&,
                const edm::InputTag&, const edm::InputTag&);

        /// destructor
        ~LogicalExpressionL1Results();

    public:

        /// return true if the logical expression is syntactically correct 
        inline bool isValid() {
            return m_validLogicalExpression;
        }

        /// update quantities related to the logical expression at the beginning of the run
        ///     new logical expression, replacing the logical expression given the in previous run
        const int logicalExpressionRunUpdate(const edm::Run&,
                const edm::EventSetup&, const std::string&);
        //
        ///     keep the logical expression given in the previous run
        const int logicalExpressionRunUpdate(const edm::Run&,
                const edm::EventSetup&);

        /// list of triggers in the logical expression, trigger decisions, prescale factors and masks, error codes

        inline const std::vector<L1GtLogicParser::OperandToken>& expL1Triggers() {
            return m_expL1Triggers;
        }
        const std::vector<std::pair<std::string, bool> >& decisionsBeforeMask();
        const std::vector<std::pair<std::string, bool> >& decisionsAfterMask();
        const std::vector<std::pair<std::string, int> >& prescaleFactors();
        const std::vector<std::pair<std::string, int> >& triggerMasks();

        const std::vector<std::pair<std::string, int> >& errorCodes(
                const edm::Event&);

    private:

        /// parse the logical expression, initialize the private members to required size 
        /// such that one can just reset them
        bool initialize();

        /// reset for each L1 trigger the value from pair.second
        void reset(std::vector<std::pair<std::string, bool> >) const;
        void reset(std::vector<std::pair<std::string, int> >) const;

        void
        l1Results(const edm::Event& iEvent,
                const edm::InputTag& l1GtRecordInputTag,
                const edm::InputTag& l1GtReadoutRecordInputTag);

    private:

        /// private members as input parameters

        /// logical expression 
        std::string m_logicalExpression;

        L1GtUtils& m_l1GtUtils;

        edm::InputTag m_l1GtRecordInputTag;
        edm::InputTag m_l1GtReadoutRecordInputTag;

    private:

        // private members

        /// code for L1 trigger configuration
        int m_l1ConfCode;

        /// true if valid L1 configuration - if not, reset all quantities and return
        bool m_validL1Configuration;

        /// true if the logical expression uses accepted L1GtLogicParser operators  
        bool m_validLogicalExpression;

        /// true if input tags for GT records are to be found from provenance 
        /// (if both input tags from constructors are empty)
        bool m_l1GtInputTagsFromProv;

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
    void retrieveL1EventSetup(const edm::EventSetup&);

    /// retrieve L1GtTriggerMenuLite (per run product) and cache it to improve the speed

    ///    for use in beginRun(const edm::Run&, const edm::EventSetup&);
    ///        input tag explicitly given
    void retrieveL1GtTriggerMenuLite(const edm::Run&, const edm::InputTag&);

    /// get all the run-constant quantities for L1 trigger and cache them

    ///    for use in beginRun(const edm::Run&, const edm::EventSetup&);
    ///        input tag for L1GtTriggerMenuLite explicitly given
    void getL1GtRunCache(const edm::Run&, const edm::EventSetup&, const bool,
            const bool, const edm::InputTag&);
    ///        input tag for L1GtTriggerMenuLite found from provenance
    void getL1GtRunCache(const edm::Run&, const edm::EventSetup&, const bool,
            const bool);

    ///    for use in analyze(const edm::Event&, const edm::EventSetup&)
    ///        input tag for L1GtTriggerMenuLite explicitly given
    void getL1GtRunCache(const edm::Event&, const edm::EventSetup&, const bool,
            const bool, const edm::InputTag&);
    ///        input tag for L1GtTriggerMenuLite found from provenance
    void getL1GtRunCache(const edm::Event&, const edm::EventSetup&, const bool,
            const bool);

    /// find from provenance the input tags for L1GlobalTriggerRecord and
    /// L1GlobalTriggerReadoutRecord
    /// if the product does not exist, return empty input tags
    void getL1GtRecordInputTag(const edm::Event& iEvent,
            edm::InputTag& l1GtRecordInputTag,
            edm::InputTag& l1GtReadoutRecordInputTag) const;

    /// get the input tag for L1GtTriggerMenuLite
    void getL1GtTriggerMenuLiteInputTag(const edm::Run& iRun,
            edm::InputTag& l1GtTriggerMenuLiteInputTag) const;
    
    /// return the input tags found from provenance
    inline const edm::InputTag& provL1GtRecordInputTag() {
        return m_provL1GtRecordInputTag;
    }
    
    inline const edm::InputTag& provL1GtReadoutRecordInputTag() {
        return m_provL1GtReadoutRecordInputTag;
    }
    
    inline const edm::InputTag& provL1GtTriggerMenuLiteInputTag() {
        return m_provL1GtTriggerMenuLiteInputTag;
    }

    /// return the trigger "category" trigCategory
    ///    algorithm trigger alias or algorithm trigger name AlgorithmTrigger = 0,
    ///    technical trigger TechnicalTrigger = 1
    /// and its bit number
    ///
    /// in case the algorithm trigger / technical trigger is not in the menu,
    /// the returned function is false, the trigger category is irrelevant
    /// (default value is AlgorithmTrigger), and the value of the bit number is -1

    const bool l1AlgoTechTrigBitNumber(const std::string& nameAlgoTechTrig,
            TriggerCategory& trigCategory, int& bitNumber) const;

    /// return the trigger name and alias for a given trigger category and a given 
    /// bit number
    ///
    /// in case no algorithm trigger / technical trigger is defined for that bit in the menu,
    /// the returned function is false, and the name and the alias is empty

    const bool l1TriggerNameFromBit(const int& bitNumber,
            const TriggerCategory& trigCategory, std::string& aliasL1Trigger,
            std::string& nameL1Trigger) const;

    /// return results for a given algorithm or technical trigger:
    /// input:
    ///   event
    ///   input tag for the L1GlobalTriggerRecord product
    ///   input tag for the L1GlobalTriggerReadoutRecord product
    ///   algorithm trigger name or alias, or technical trigger name
    /// output (by reference):
    ///    decision before mask,
    ///    decision after mask,
    ///    prescale factor
    ///    trigger mask
    /// return: integer error code

    const int
            l1Results(const edm::Event& iEvent,
                    const edm::InputTag& l1GtRecordInputTag,
                    const edm::InputTag& l1GtReadoutRecordInputTag,
                    const std::string& nameAlgoTechTrig,
                    bool& decisionBeforeMask, bool& decisionAfterMask,
                    int& prescaleFactor, int& triggerMask) const;

    /// return results for a given algorithm or technical trigger,
    /// input tag for the an appropriate EDM product will be found from provenance
    /// input:
    ///   event
    ///   algorithm trigger name or alias, or technical trigger name
    /// output (by reference):
    ///    decision before mask,
    ///    decision after mask,
    ///    prescale factor
    ///    trigger mask
    /// return: integer error code

    const int
            l1Results(const edm::Event& iEvent,
                    const std::string& nameAlgoTechTrig,
                    bool& decisionBeforeMask, bool& decisionAfterMask,
                    int& prescaleFactor, int& triggerMask) const;

    /// for the functions decisionBeforeMask, decisionAfterMask, decision
    /// prescaleFactor, trigger mask:
    ///
    /// input:
    ///   event, event setup
    ///   input tag for the L1GlobalTriggerRecord product
    ///   input tag for the L1GlobalTriggerReadoutRecord product
    ///   algorithm trigger name or alias, or technical trigger name
    /// output (by reference):
    ///    error code
    /// return: the corresponding quantity
    ///
    /// if input tags are not given, they are found for the appropriate EDM products
    /// from provenance

    ///   return decision before trigger mask for a given algorithm or technical trigger
    const bool decisionBeforeMask(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const std::string& nameAlgoTechTrig, int& errorCode) const;

    const bool decisionBeforeMask(const edm::Event& iEvent,
            const std::string& nameAlgoTechTrig, int& errorCode) const;

    ///   return decision after trigger mask for a given algorithm or technical trigger
    const bool decisionAfterMask(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const std::string& nameAlgoTechTrig, int& errorCode) const;

    const bool decisionAfterMask(const edm::Event& iEvent,
            const std::string& nameAlgoTechTrig, int& errorCode) const;

    ///   return decision after trigger mask for a given algorithm or technical trigger
    ///          function identical with decisionAfterMask
    const bool decision(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const std::string& nameAlgoTechTrig, int& errorCode) const;

    const bool decision(const edm::Event& iEvent,
            const std::string& nameAlgoTechTrig, int& errorCode) const;

    ///   return prescale factor for a given algorithm or technical trigger
    const int prescaleFactor(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const std::string& nameAlgoTechTrig, int& errorCode) const;

    const int prescaleFactor(const edm::Event& iEvent,
            const std::string& nameAlgoTechTrig, int& errorCode) const;

    ///   return trigger mask for a given algorithm or technical trigger
    const int triggerMask(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const std::string& nameAlgoTechTrig, int& errorCode) const;

    const int triggerMask(const edm::Event& iEvent,
            const std::string& nameAlgoTechTrig, int& errorCode) const;

    ///     faster than previous two methods - one needs in fact for the
    ///     masks the event setup only
    const int
            triggerMask(const std::string& nameAlgoTechTrig, int& errorCode) const;

    /// return the index of the actual set of prescale factors used for the
    /// event (must be the same for all events in the luminosity block,
    /// if no errors)
    ///

    const int prescaleFactorSetIndex(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const TriggerCategory& trigCategory, int& errorCode) const;

    const int prescaleFactorSetIndex(const edm::Event& iEvent,
            const TriggerCategory& trigCategory, int& errorCode) const;


    /// return the actual set of prescale factors used for the
    /// event (must be the same for all events in the luminosity block,
    /// if no errors)

    const std::vector<int>& prescaleFactorSet(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const TriggerCategory& trigCategory, int& errorCode);

    const std::vector<int>& prescaleFactorSet(const edm::Event& iEvent,
            const TriggerCategory& trigCategory, int& errorCode);


    /// return the set of trigger masks for the physics partition (partition zero)
    /// used for the event (remain the same in the whole run, if no errors)
    const std::vector<unsigned int>& triggerMaskSet(
            const TriggerCategory& trigCategory, int& errorCode);


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
    const bool trigResult(const DecisionWord& decWord, const int bitNumber,
            const std::string& nameAlgoTechTrig,
            const TriggerCategory& trigCategory, int& errorCode) const;

private:

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

    /// cached input tags from provenance - they are updated once per run only

    mutable edm::InputTag m_provL1GtRecordInputTag;
    mutable edm::InputTag m_provL1GtReadoutRecordInputTag;
    mutable edm::InputTag m_provL1GtTriggerMenuLiteInputTag;
    
    /// run cache ID 
    edm::RunID m_runIDCache;
    edm::RunID m_provRunIDCache;


private:

    /// index of physics DAQ partition
    unsigned int m_physicsDaqPartition;

    std::vector<unsigned int> m_triggerMaskSet;
    std::vector<int> m_prescaleFactorSet;

    /// flags to check which method was used to retrieve L1 trigger configuration
    bool m_retrieveL1EventSetup;
    bool m_retrieveL1GtTriggerMenuLite;

};

#endif /*GlobalTriggerAnalyzer_L1GtUtils_h*/
