#ifndef GlobalTriggerAnalyzer_L1GtTrigReport_h
#define GlobalTriggerAnalyzer_L1GtTrigReport_h

/**
 * \class L1GtTrigReport
 * 
 * 
 * Description: L1 Trigger report.  
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

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// class declaration

class L1GtTrigReport : public edm::EDAnalyzer
{

public:

    /// constructor
    explicit L1GtTrigReport(const edm::ParameterSet&);

    /// destructor
    virtual ~L1GtTrigReport();

private:

    virtual void beginJob(const edm::EventSetup&) ;

    /// analyze each event
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    /// end of job
    virtual void endJob() ;

private:

    /// temporary: get map from algorithm names to algorithm bits
    void getAlgoMap(const edm::Event&, const edm::EventSetup&);

private:

    /// input tag for GT readout collection (DAQ record):
    edm::InputTag m_l1GtDaqInputTag;

    /// inputTag for L1 Global Trigger object maps
    edm::InputTag m_l1GtObjectMapTag;

    /// print verbosity
    int m_printVerbosity;

    /// print output
    int m_printOutput;


    /// counters

    /// number of events processed
    int  m_totalEvents;

    /// number of events with error (EDProduct[s] not found)
    int  m_nErrors;

    /// number of events accepted by any of the L1 algorithm
    int  m_globalAccepts;

    /// number of events accepted by each L1 algorithm
    /// map<algorithm name, count>
    std::map<std::string, int> m_nAlgoAccepts;

    /// number of events rejected by each L1 algorithm
    /// map<algorithm name, count>
    std::map<std::string, int> m_nAlgoRejects;

    /// prescale factor
    /// map<algorithm name, factor>
    std::map<std::string, int> m_prescaleFactor;

    /// trigger mask
    /// map<algorithm name, mask>
    std::map<std::string, unsigned int> m_triggerMask;

    // temporary TODO change after trigger menu available as EventSetup

    std::map<int, std::string> m_algoBitToName;
    bool m_algoMap;

};

#endif /*GlobalTriggerAnalyzer_L1GtTrigReport_h*/
