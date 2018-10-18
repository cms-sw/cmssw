/**
 * \class L1GtPsbSetupTrivialProducer
 *
 *
 * Description: ESProducer for the setup of L1 GT PSB boards.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtPsbSetupTrivialProducer.h"

// system include files
#include <memory>

#include <vector>

// user include files
#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1GtPsbSetupRcd.h"

// forward declarations

//


// constructor(s)
L1GtPsbSetupTrivialProducer::L1GtPsbSetupTrivialProducer(
        const edm::ParameterSet& parSet)
{
    // tell the framework what data is being produced
    setWhatProduced(this, &L1GtPsbSetupTrivialProducer::producePsbSetup);

    // now do what ever other initialization is needed

    // detailed input configuration for PSB
    std::vector<edm::ParameterSet> psbSetup = parSet.getParameter<std::vector<
            edm::ParameterSet> > ("PsbSetup");

    // reserve space for L1 GT boards
    m_gtPsbSetup.reserve(psbSetup.size());

    std::vector<unsigned int> enableRecLvdsInt;
    enableRecLvdsInt.reserve(L1GtPsbConfig::PsbNumberLvdsGroups);
    std::vector<bool> enableRecLvds;
    enableRecLvds.reserve(L1GtPsbConfig::PsbNumberLvdsGroups);

    std::vector<unsigned int> enableRecSerLinkInt;
    enableRecSerLinkInt.reserve(L1GtPsbConfig::PsbSerLinkNumberChannels);
    std::vector<bool> enableRecSerLink;
    enableRecSerLink.reserve(L1GtPsbConfig::PsbSerLinkNumberChannels);

    for (std::vector<edm::ParameterSet>::const_iterator itPSet =
            psbSetup.begin(); itPSet != psbSetup.end(); ++itPSet) {

        //
        L1GtPsbConfig psbConfig(itPSet->getParameter<int> ("Slot"));

        psbConfig.setGtPsbCh0SendLvds(
                itPSet->getParameter<bool> ("Ch0SendLvds"));
        psbConfig.setGtPsbCh1SendLvds(
                itPSet->getParameter<bool> ("Ch1SendLvds"));

        enableRecLvdsInt = itPSet->getParameter<std::vector<unsigned int> > (
                "EnableRecLvds");

        for (std::vector<unsigned int>::const_iterator cIt =
                enableRecLvdsInt.begin(); cIt != enableRecLvdsInt.end(); ++cIt) {
            bool val = *cIt;
            enableRecLvds.push_back(val);
        }

        psbConfig.setGtPsbEnableRecLvds(enableRecLvds);
        enableRecLvds.clear();

        enableRecSerLinkInt
                = itPSet->getParameter<std::vector<unsigned int> > (
                        "EnableRecSerLink");

        for (std::vector<unsigned int>::const_iterator cIt =
                enableRecSerLinkInt.begin(); cIt != enableRecSerLinkInt.end(); ++cIt) {
            bool val = *cIt;
            enableRecSerLink.push_back(val);
        }

        psbConfig.setGtPsbEnableRecSerLink(enableRecSerLink);
        enableRecSerLink.clear();

        // push the board in the vector
        m_gtPsbSetup.push_back(psbConfig);

    }

}

// destructor
L1GtPsbSetupTrivialProducer::~L1GtPsbSetupTrivialProducer()
{

    // empty

}

// member functions

// method called to produce the data
std::unique_ptr<L1GtPsbSetup> L1GtPsbSetupTrivialProducer::producePsbSetup(
        const L1GtPsbSetupRcd& iRecord)
{
    auto pL1GtPsbSetup = std::make_unique<L1GtPsbSetup>();

    pL1GtPsbSetup->setGtPsbSetup(m_gtPsbSetup);

    return pL1GtPsbSetup;
}
