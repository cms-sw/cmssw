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
 *
 */

// system include files
#include <memory>
#include <string>
#include <vector>
#include <list>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtTrigReportEntry.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerRecord.h"


// forward declarations
class L1GtStableParameters;
class L1GtPrescaleFactors;
class L1GtTriggerMask;
class L1GtTriggerMenu;

// class declaration

class L1GtTrigReport : public edm::one::EDAnalyzer<>
{

public:

    /// constructor
    explicit L1GtTrigReport(const edm::ParameterSet&);

    /// destructor
    virtual ~L1GtTrigReport();

private:

    virtual void beginJob();

    /// analyze each event
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    /// end of job
    virtual void endJob();

private:

    /// cached stuff

    /// stable parameters
    const L1GtStableParameters* m_l1GtStablePar;
    unsigned long long m_l1GtStableParCacheID;

    /// number of physics triggers
    unsigned int m_numberPhysTriggers;

    /// number of technical triggers
    unsigned int m_numberTechnicalTriggers;

    /// number of DAQ partitions
    unsigned int m_numberDaqPartitions;
    unsigned int m_numberDaqPartitionsMax;

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


private:

    /// boolean flag to select the input record
    const bool m_useL1GlobalTriggerRecord;

    /// input tag for GT record (L1 GT DAQ record or L1 GT "lite" record):
    const edm::InputTag m_l1GtRecordInputTag;

    const edm::EDGetTokenT<L1GlobalTriggerRecord> m_l1GtRecordInputToken1;
    const edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> m_l1GtRecordInputToken2;

    /// print verbosity
    const int m_printVerbosity;

    /// print output
    const int m_printOutput;

    /// counters

    /// global number of events processed
    int m_totalEvents;

    /// global number of events with error (EDProduct[s] not found)
    std::vector<int> m_globalNrErrors;

    /// global number of events accepted by any of the L1 algorithm in any menu
    std::vector<int> m_globalNrAccepts;

    /// list of individual entries in the report for physics algorithms
    std::list<L1GtTrigReportEntry*> m_entryList;

    /// list of individual entries in the report for technical triggers
    std::list<L1GtTrigReportEntry*> m_entryListTechTrig;

    typedef std::list<L1GtTrigReportEntry*>::const_iterator CItEntry;
    typedef std::list<L1GtTrigReportEntry*>::iterator ItEntry;

    /// index of physics DAQ partition
    const unsigned int m_physicsDaqPartition;

};

#endif /*GlobalTriggerAnalyzer_L1GtTrigReport_h*/
