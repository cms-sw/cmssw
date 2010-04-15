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

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

// forward declarations
class L1GtStableParameters;
class L1GtPrescaleFactors;
class L1GtTriggerMask;
class L1GtTriggerMenu;

// class declaration

class L1GtUtils {

public:

    /// constructor
    explicit L1GtUtils();

    /// destructor
    virtual ~L1GtUtils();

public:

    /// public methods


    /// retrieve all the relevant L1 trigger event setup records and cache them to improve the speed
    void retrieveL1EventSetup(const edm::EventSetup&);

    /// get the input tags for L1GlobalTriggerRecord and L1GlobalTriggerReadoutRecord
    void getInputTag(const edm::Event& iEvent, edm::InputTag& l1GtRecordInputTag,
            edm::InputTag& l1GtReadoutRecordInputTag) const;

    /// return the trigger "algorithm type" triggerAlgTechTrig
    ///    physics algorithm alias or physics algorithm name = 0,
    ///    technical trigger = 1
    /// and its bit number
    ///
    /// in case the physics algorithm / technical trigger is not in the menu, the returned function is false,
    /// the trigger "algorithm type" is -1, and the value of the bit number is -1

    const bool l1AlgTechTrigBitNumber(const std::string& nameAlgTechTrig,
            int& triggerAlgTechTrig, int& bitNumber) const;

    /// return results for a given algorithm or technical trigger:
    /// input:
    ///   event
    ///   input tag for the L1GlobalTriggerRecord product
    ///   input tag for the L1GlobalTriggerReadoutRecord product
    ///   physics algorithm name or alias or technical trigger name
    /// output (by reference):
    ///    decision before mask,
    ///    decision after mask,
    ///    prescale factor
    ///    trigger mask
    /// return: integer error code

    const int l1Results(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const std::string& nameAlgTechTrig, bool& decisionBeforeMask,
            bool& decisionAfterMask, int& prescaleFactor, int& triggerMask) const;

    /// return results for a given algorithm or technical trigger,
    /// input tag for the an appropriate EDM product will be found from provenance
    /// input:
    ///   event
    ///   physics algorithm name or alias or technical trigger name
    /// output (by reference):
    ///    decision before mask,
    ///    decision after mask,
    ///    prescale factor
    ///    trigger mask
    /// return: integer error code

    const int l1Results(const edm::Event& iEvent,
            const std::string& nameAlgTechTrig, bool& decisionBeforeMask,
            bool& decisionAfterMask, int& prescaleFactor, int& triggerMask) const;

    /// for the functions decisionBeforeMask, decisionAfterMask, decision
    /// prescaleFactor, trigger mask:
    ///
    /// input:
    ///   event, event setup
    ///   input tag for the L1GlobalTriggerRecord product
    ///   input tag for the L1GlobalTriggerReadoutRecord product
    ///   physics algorithm name or alias or technical trigger name
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
            const std::string& nameAlgTechTrig, int& errorCode) const;

    const bool decisionBeforeMask(const edm::Event& iEvent,
            const std::string& nameAlgTechTrig, int& errorCode) const;


    ///   return decision after trigger mask for a given algorithm or technical trigger
    const bool decisionAfterMask(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const std::string& nameAlgTechTrig, int& errorCode) const;

    const bool decisionAfterMask(const edm::Event& iEvent,
            const std::string& nameAlgTechTrig, int& errorCode) const;


    ///   return decision after trigger mask for a given algorithm or technical trigger
    ///          function identical with decisionAfterMask
    const bool decision(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const std::string& nameAlgTechTrig, int& errorCode) const;

    const bool decision(const edm::Event& iEvent,
            const std::string& nameAlgTechTrig, int& errorCode) const;

    ///   return prescale factor for a given algorithm or technical trigger
    const int prescaleFactor(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const std::string& nameAlgTechTrig, int& errorCode) const;

    const int prescaleFactor(const edm::Event& iEvent,
            const std::string& nameAlgTechTrig, int& errorCode) const;

    ///   return trigger mask for a given algorithm or technical trigger
    const int triggerMask(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const std::string& nameAlgTechTrig, int& errorCode) const;

    const int triggerMask(const edm::Event& iEvent,
            const std::string& nameAlgTechTrig, int& errorCode) const;

    ///     faster than previous two methods - one needs in fact for the
    ///     masks the event setup only
    const int triggerMask(const std::string& nameAlgTechTrig, int& errorCode) const;

    /// return the index of the actual set of prescale factors used for the
    /// event (must be the same for all events in the luminosity block,
    /// if no errors)
    const int prescaleFactorSetIndex(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const std::string& triggerAlgTechTrig, int& errorCode) const;

    const int prescaleFactorSetIndex(const edm::Event& iEvent,
            const std::string& triggerAlgTechTrig, int& errorCode) const;

    /// return the actual set of prescale factors used for the
    /// event (must be the same for all events in the luminosity block,
    /// if no errors)
    const std::vector<int>& prescaleFactorSet(const edm::Event& iEvent,
            const edm::InputTag& l1GtRecordInputTag,
            const edm::InputTag& l1GtReadoutRecordInputTag,
            const std::string& triggerAlgTechTrig, int& errorCode);

    const std::vector<int>& prescaleFactorSet(const edm::Event& iEvent,
            const std::string& triggerAlgTechTrig, int& errorCode);


    /// return the set of trigger masks for the physics partition (partition zero)
    /// used for the event (remain the same in the whole run, if no errors)
    const std::vector<unsigned int>& triggerMaskSet(
            const std::string& triggerAlgTechTrig, int& errorCode);

    /// return the L1 trigger menu name
    const std::string& l1TriggerMenu() const;

    /// return the L1 trigger menu implementation
    const std::string& l1TriggerMenuImplementation() const;

private:

    /// cached stuff

    /// stable parameters
    const L1GtStableParameters* m_l1GtStablePar;
    unsigned long long m_l1GtStableParCacheID;

    /// number of physics triggers
    unsigned int m_numberPhysTriggers;

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

    std::vector<unsigned int> m_triggerMaskAlgoTrig;
    std::vector<unsigned int> m_triggerMaskTechTrig;

    std::vector<unsigned int> m_triggerMaskVetoAlgoTrig;
    std::vector<unsigned int> m_triggerMaskVetoTechTrig;

    // trigger menu
    const L1GtTriggerMenu* m_l1GtMenu;
    unsigned long long m_l1GtMenuCacheID;

    const AlgorithmMap* m_algorithmMap;
    const AlgorithmMap* m_algorithmAliasMap;
    const AlgorithmMap* m_technicalTriggerMap;

private:

    /// index of physics DAQ partition
    unsigned int m_physicsDaqPartition;

    std::vector<unsigned int> m_triggerMaskSet;
    std::vector<int> m_prescaleFactorSet;


};

#endif /*GlobalTriggerAnalyzer_L1GtUtils_h*/
