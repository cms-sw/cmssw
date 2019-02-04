#ifndef L1GtConfigProducers_L1GtPsbSetupTrivialProducer_h
#define L1GtConfigProducers_L1GtPsbSetupTrivialProducer_h

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

// system include files
#include <memory>

#include <vector>

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/L1TObjects/interface/L1GtPsbConfig.h"
#include "CondFormats/L1TObjects/interface/L1GtPsbSetup.h"

// forward declarations
class L1GtPsbSetupRcd;

// class declaration
class L1GtPsbSetupTrivialProducer : public edm::ESProducer
{

public:

    /// constructor
    L1GtPsbSetupTrivialProducer(const edm::ParameterSet&);

    /// destructor
    ~L1GtPsbSetupTrivialProducer() override;


    /// public methods

    /// produce the setup for L1 GT PSB boards
    std::unique_ptr<L1GtPsbSetup> producePsbSetup(
        const L1GtPsbSetupRcd&);

private:

    /// L1 GT PSB boards and their setup
    std::vector<L1GtPsbConfig> m_gtPsbSetup;

};

#endif
