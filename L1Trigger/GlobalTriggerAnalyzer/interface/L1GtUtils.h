#ifndef GlobalTriggerAnalyzer_L1GtUtils_h
#define GlobalTriggerAnalyzer_L1GtUtils_h

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

// system include files
#include <string>

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

/// get the result for a given algorithm via trigger menu from the L1 GT lite record
bool l1AlgorithmResult(const edm::Event&, const edm::EventSetup&,
        const edm::InputTag&, const std::string&);

/// get the result for a given algorithm via trigger menu from the L1 GT lite record
/// assume InputTag for L1GlobalTriggerRecord to be l1GtRecord
bool l1AlgorithmResult(const edm::Event&, const edm::EventSetup&,
        const std::string&);

#endif /*GlobalTriggerAnalyzer_L1GtUtils_h*/
