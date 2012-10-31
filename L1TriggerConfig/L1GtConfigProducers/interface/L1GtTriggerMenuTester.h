#ifndef L1GtConfigProducers_L1GtTriggerMenuTester_h
#define L1GtConfigProducers_L1GtTriggerMenuTester_h

/**
 * \class L1GtTriggerMenuTester
 * 
 * 
 * Description: test analyzer for L1 GT trigger menu.  
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
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtTriggerMenuTester.h"

// system include files
#include <string>
#include <map>

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/L1TObjects/interface/L1GtAlgorithm.h"

// forward declarations


// class declaration
class L1GtTriggerMenuTester: public edm::EDAnalyzer {

public:

    // constructor
    explicit L1GtTriggerMenuTester(const edm::ParameterSet&);

    // destructor
    virtual ~L1GtTriggerMenuTester();

    virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:

    /// constant iterator
    typedef std::map<std::string, const L1GtAlgorithm*>::const_iterator
            CItAlgoP;

    /// name of HTML file attached to the wiki page
    std::string m_htmlFile;

    /// printing template for a trigger group
    void printTriggerGroup(const std::string& trigGroupName,
            const std::map<std::string, const L1GtAlgorithm*>& trigGroup);

};

#endif /*L1GtConfigProducers_L1GtTriggerMenuTester_h*/
