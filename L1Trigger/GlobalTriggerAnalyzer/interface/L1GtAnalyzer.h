#ifndef GlobalTriggerAnalyzer_L1GtAnalyzer_h
#define GlobalTriggerAnalyzer_L1GtAnalyzer_h

/**
 * \class L1GtAnalyzer
 * 
 * 
 * Description: test analyzer to illustrate various methods for L1 GT trigger.
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
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

// class declaration

class L1GtAnalyzer: public edm::EDAnalyzer {

public:
    explicit L1GtAnalyzer(const edm::ParameterSet&);
    ~L1GtAnalyzer();

private:

    virtual void beginJob();
    virtual void beginRun(const edm::Run&, const edm::EventSetup&);
    virtual void beginLuminosityBlock(const edm::LuminosityBlock&,
            const edm::EventSetup&);

    /// analyze: decision and decision word
    ///   bunch cross in event BxInEvent = 0 - L1Accept event
    virtual void analyzeDecisionReadoutRecord(const edm::Event&, const edm::EventSetup&);

    /// analyze: decision for a given algorithm via trigger menu
    void analyzeDecisionLiteRecord(const edm::Event&, const edm::EventSetup&);

    /// analyze: usage of L1GtUtils
    void analyzeL1GtUtilsCore(const edm::Event&, const edm::EventSetup&);
    ///   for tests, use only one of the following methods
    void analyzeL1GtUtilsMenuLite(const edm::Event&, const edm::EventSetup&);
    void analyzeL1GtUtilsEventSetup(const edm::Event&, const edm::EventSetup&);
    void analyzeL1GtUtils(const edm::Event&, const edm::EventSetup&);

    /// analyze: object map product
    virtual void analyzeObjectMap(const edm::Event&, const edm::EventSetup&);

    /// analyze: usage of L1GtTriggerMenuLite
    void analyzeL1GtTriggerMenuLite(const edm::Event&, const edm::EventSetup&);

    /// analyze: usage of ConditionsInEdm
    ///
    /// to be used in beginRun
    void analyzeConditionsInRunBlock(const edm::Run&, const edm::EventSetup&);
    /// to be used in beginLuminosityBlock
    void analyzeConditionsInLumiBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
    /// to be used in analyze/produce/filter
    void analyzeConditionsInEventBlock(const edm::Event&, const edm::EventSetup&);

    /// analyze each event: event loop over various code snippets
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    /// end section
    virtual void endLuminosityBlock(const edm::LuminosityBlock&,
            const edm::EventSetup&);
    virtual void endRun(const edm::Run&, const edm::EventSetup&);

    /// end of job
    virtual void endJob();

private:

    /// input tags for GT DAQ product
    edm::InputTag m_l1GtDaqReadoutRecordInputTag;

    /// input tags for GT lite product
    edm::InputTag m_l1GtRecordInputTag;

    /// input tags for GT object map collection
    edm::InputTag m_l1GtObjectMapTag;

    /// input tag for muon collection from GMT
    edm::InputTag m_l1GmtInputTag;

    /// input tag for L1GtTriggerMenuLite
    edm::InputTag m_l1GtTmLInputTag;

    /// input tag for ConditionInEdm products
    edm::InputTag m_condInEdmInputTag;

    /// a physics algorithm (name or alias) or a technical trigger name
    std::string m_nameAlgTechTrig;

    /// a condition in the physics algorithm to test the object maps
    std::string m_condName;

    /// a bit number to retrieve the name and the alias
    unsigned int m_bitNumber;

    /// L1 configuration code for L1GtUtils
    unsigned int m_l1GtUtilsConfiguration;

    /// if true, use methods in L1GtUtils with the input tag for L1GtTriggerMenuLite
    /// from provenance
    bool m_l1GtTmLInputTagProv;

    /// if true, configure (partially) L1GtUtils in beginRun using getL1GtRunCache
    bool m_l1GtUtilsConfigureBeginRun;

private:

    L1GtUtils m_l1GtUtils;

};

#endif /*GlobalTriggerAnalyzer_L1GtAnalyzer_h*/
