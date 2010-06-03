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
        AlgorithmTrigger = 0,
        TechnicalTrigger = 1
    };

    /// public methods

    // enum to string for TriggerCategory
    const std::string triggerCategory (const TriggerCategory&) const;

    /// retrieve all the relevant L1 trigger event setup records and cache them to improve the speed
    void retrieveL1EventSetup(const edm::EventSetup&);

    /// retrieve L1GtTriggerMenuLite (per run product) and cache it to improve the speed
    ///    input tag explicitly given
    void retrieveL1GtTriggerMenuLite(const edm::Event&, edm::InputTag&);
    ///    input tag found from provenance
    void retrieveL1GtTriggerMenuLite(const edm::Event&);

    /// get the input tags for L1GlobalTriggerRecord and L1GlobalTriggerReadoutRecord
    void getInputTag(const edm::Event& iEvent, edm::InputTag& l1GtRecordInputTag,
            edm::InputTag& l1GtReadoutRecordInputTag) const;

    /// get the input tags for L1GtTriggerMenuLite
    void getL1GtTriggerMenuLiteInputTag(const edm::Event& iEvent,
            edm::InputTag& l1GtTriggerMenuLiteInputTag) const;

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

    /// deprecated version - use l1AlgoTechTrigBitNumber
    const bool l1AlgTechTrigBitNumber(const std::string& nameAlgoTechTrig,
            int& triggerAlgoTechTrig, int& bitNumber) const;

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

    const int l1Results(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const std::string& nameAlgoTechTrig, bool& decisionBeforeMask,
            bool& decisionAfterMask, int& prescaleFactor, int& triggerMask) const;

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

    const int l1Results(const edm::Event& iEvent,
            const std::string& nameAlgoTechTrig, bool& decisionBeforeMask,
            bool& decisionAfterMask, int& prescaleFactor, int& triggerMask) const;

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
    const int triggerMask(const std::string& nameAlgoTechTrig, int& errorCode) const;

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

    ///     deprecated versions
    const int prescaleFactorSetIndex(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const std::string& triggerAlgoTechTrig, int& errorCode) const;

    const int prescaleFactorSetIndex(const edm::Event& iEvent,
            const std::string& triggerAlgoTechTrig, int& errorCode) const;

    /// return the actual set of prescale factors used for the
    /// event (must be the same for all events in the luminosity block,
    /// if no errors)

    const std::vector<int>& prescaleFactorSet(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const TriggerCategory& trigCategory, int& errorCode);

    const std::vector<int>& prescaleFactorSet(const edm::Event& iEvent,
            const TriggerCategory& trigCategory, int& errorCode);

    ///     deprecated versions
    const std::vector<int>& prescaleFactorSet(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const std::string& triggerAlgoTechTrig, int& errorCode);

    const std::vector<int>& prescaleFactorSet(const edm::Event& iEvent,
            const std::string& triggerAlgoTechTrig, int& errorCode);


    /// return the set of trigger masks for the physics partition (partition zero)
    /// used for the event (remain the same in the whole run, if no errors)
    const std::vector<unsigned int>& triggerMaskSet(
            const TriggerCategory& trigCategory, int& errorCode);


    ///     deprecated version
    const std::vector<unsigned int>& triggerMaskSet(
            const std::string& triggerAlgoTechTrig, int& errorCode);

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

    const edm::RunID* m_runIDCache;

    //
    const edm::RunID* m_provRunIDCache;

    //
    bool m_l1GtMenuLiteValid;

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
