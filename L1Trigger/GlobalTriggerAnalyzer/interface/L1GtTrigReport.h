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
#include <string>
#include <list>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtTrigReportEntry.h"

// class declaration

class L1GtTrigReport : public edm::EDAnalyzer
{

public:

    /// constructor
    explicit L1GtTrigReport(const edm::ParameterSet&);

    /// destructor
    virtual ~L1GtTrigReport();

private:

    virtual void beginJob(const edm::EventSetup&);

    /// analyze each event
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    /// end of job
    virtual void endJob();

private:

    /// input tag for GT readout collection (DAQ record):
    edm::InputTag m_l1GtDaqInputTag;

    /// print verbosity
    int m_printVerbosity;

    /// print output
    int m_printOutput;

    /// counters

    /// global number of events processed
    int m_totalEvents;

    /// global number of events with error (EDProduct[s] not found)
    int m_globalNrErrors;

    /// global number of events accepted by any of the L1 algorithm in any menu
    int m_globalNrAccepts;

    /// list of individual entries in the report    
    std::list<L1GtTrigReportEntry*> m_entryList;
    typedef std::list<L1GtTrigReportEntry*>::const_iterator CItEntry;
    typedef std::list<L1GtTrigReportEntry*>::iterator ItEntry;
    
    

};

#endif /*GlobalTriggerAnalyzer_L1GtTrigReport_h*/
