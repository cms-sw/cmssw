#ifndef GlobalTriggerAnalyzer_L1GtAnalyzer_h
#define GlobalTriggerAnalyzer_L1GtAnalyzer_h

/**
 * \class L1GtAnalyzer
 * 
 * 
 * Description: example for L1 Global Trigger analyzer.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date:$
 * $Revision:$
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// class declaration

class L1GtAnalyzer : public edm::EDAnalyzer
{

public:
    explicit L1GtAnalyzer(const edm::ParameterSet&);
    ~L1GtAnalyzer();

private:

    virtual void beginJob(const edm::EventSetup&) ;

    /// analyze: decision and decision word
    ///   bunch cross in event BxInEvent = 0 - L1Accept event
    virtual void analyzeDecision(const edm::Event&, const edm::EventSetup&);

    /// analyze: test setting decision
    ///   bunch cross in event BxInEvent = 0 - L1Accept event
    virtual void analyzeSetDecision(const edm::Event&, const edm::EventSetup&);

    /// print/access L1 objects in bunch cross with L1A
    virtual void analyzeL1Objects(const edm::Event&, const edm::EventSetup&);

    /// test muon part in L1GlobalTriggerReadoutRecord
    virtual void analyzeMuons(const edm::Event&, const edm::EventSetup&);

    /// analyze: object map record
    virtual void analyzeObjectMap(const edm::Event&, const edm::EventSetup&);

    /// analyze: seed muon HLT using the object map record
    virtual void analyzeObjectMapMuons(const edm::Event&, const edm::EventSetup&);

    /// analyze each event: event loop over various code snippets
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    /// end of job
    virtual void endJob() ;

private:

    static const edm::ParameterSet* m_pSet;

    // input tags for GT DAQ record
    edm::InputTag m_daqGtInputTag;

};

#endif /*GlobalTriggerAnalyzer_L1GtAnalyzer_h*/
