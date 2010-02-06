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
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

// class declaration

class L1GtAnalyzer: public edm::EDAnalyzer {

public:
    explicit L1GtAnalyzer(const edm::ParameterSet&);
    ~L1GtAnalyzer();

private:

    virtual void beginJob();

    /// analyze: decision and decision word
    ///   bunch cross in event BxInEvent = 0 - L1Accept event
    virtual void analyzeDecisionReadoutRecord(const edm::Event&, const edm::EventSetup&);

    /// analyze: decision for a given algorithm via trigger menu
    void analyzeDecisionLiteRecord(const edm::Event&, const edm::EventSetup&);

    /// analyze: usage of L1GtUtils
    void analyzeL1GtUtils(const edm::Event&, const edm::EventSetup&);

    /// analyze: object map record
    virtual void analyzeObjectMap(const edm::Event&, const edm::EventSetup&);

    /// analyze each event: event loop over various code snippets
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    /// end of job
    virtual void endJob();

private:

    /// input tags for GT DAQ record
    edm::InputTag m_l1GtDaqReadoutRecordInputTag;

    /// input tags for GT lite record
    edm::InputTag m_l1GtRecordInputTag;

    /// input tags for GT object map collection
    edm::InputTag m_l1GtObjectMapTag;

    // input tag for muon collection from GMT
    edm::InputTag m_l1GmtInputTag;

    /// a physics algorithm (name or alias) or a technical trigger name
    std::string m_nameAlgTechTrig;

    /// a condition in the physics algorithm to test the object maps
    std::string m_condName;

    L1GtUtils m_l1GtUtils;

};

#endif /*GlobalTriggerAnalyzer_L1GtAnalyzer_h*/
