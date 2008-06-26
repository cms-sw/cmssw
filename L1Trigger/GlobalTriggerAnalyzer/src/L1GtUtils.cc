/**
 * \class L1GtUtils
 * 
 * 
 * Description: various methods for L1 GT, to be called in an EDM analyzer or filter.  
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

// this class header
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

// system include files
#include <iomanip>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// get the result for a given algorithm via trigger menu from the L1 GT lite record
bool l1AlgorithmResult(const edm::Event& iEvent,
        const edm::EventSetup& evSetup,
        const edm::InputTag& l1GtRecordInputTag,
        const std::string& l1AlgorithmName) {

    edm::Handle<L1GlobalTriggerRecord> gtRecord;
    iEvent.getByLabel(l1GtRecordInputTag, gtRecord);

    if (!gtRecord.isValid()) {

        edm::LogError("L1GtUtils") << "\nL1GlobalTriggerRecord with \n  "
                << l1GtRecordInputTag << "\nnot found"
            "\n  --> returning false by default!\n" << std::endl;

        return false;

    }

    const DecisionWord gtDecisionWord = gtRecord->decisionWord();

    edm::ESHandle<L1GtTriggerMenu> l1GtMenu;
    evSetup.get<L1GtTriggerMenuRcd>().get(l1GtMenu) ;
    const L1GtTriggerMenu* m_l1GtMenu = l1GtMenu.product();

    const bool algResult = m_l1GtMenu->gtAlgorithmResult(l1AlgorithmName,
            gtDecisionWord);

    edm::LogVerbatim("L1GtUtils") << "\nResult for algorithm "
            << l1AlgorithmName << ": " << algResult << "\n" << std::endl;

    return algResult;

}

// get the result for a given algorithm via trigger menu from the L1 GT lite record
// assume InputTag for L1GlobalTriggerRecord to be l1GtRecord
bool l1AlgorithmResult(const edm::Event& iEvent,
        const edm::EventSetup& evSetup, const std::string& l1AlgorithmName) {

    typedef std::vector< edm::Provenance const*> Provenances;
    Provenances provenances;
    std::string friendlyName;
    std::string modLabel;
    std::string instanceName;
    std::string processName;
    
    edm::InputTag l1GtRecordInputTag;
    
    iEvent.getAllProvenance(provenances);

    //edm::LogVerbatim("L1GtUtils") << "\n" << "Event contains "
    //        << provenances.size() << " product" << (provenances.size()==1 ? "" : "s")
    //        << " with friendlyClassName, moduleLabel, productInstanceName and processName:"
    //        << std::endl;

    for (Provenances::iterator itProv = provenances.begin(), itProvEnd =
            provenances.end(); itProv != itProvEnd; ++itProv) {
        
        friendlyName = (*itProv)->friendlyClassName();
        modLabel = (*itProv)->moduleLabel();
        instanceName = (*itProv)->productInstanceName();
        processName = (*itProv)->processName();

        //edm::LogVerbatim("L1GtUtils") << friendlyName << " \"" << modLabel
        //        << "\" \"" << instanceName << "\" \"" << processName << "\""
        //        << std::endl;

        if (friendlyName == "L1GlobalTriggerRecord") {
            l1GtRecordInputTag = edm::InputTag(modLabel, instanceName,
                    processName);
        }
    }

    edm::LogVerbatim("L1GtUtils")
            << "\nL1GlobalTriggerRecord found in the event with \n  "
            << l1GtRecordInputTag << std::endl;

    return l1AlgorithmResult(iEvent, evSetup, l1GtRecordInputTag,
            l1AlgorithmName);

}

