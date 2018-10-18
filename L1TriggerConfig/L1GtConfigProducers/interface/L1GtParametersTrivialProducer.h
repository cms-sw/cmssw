#ifndef L1GtConfigProducers_L1GtParametersTrivialProducer_h
#define L1GtConfigProducers_L1GtParametersTrivialProducer_h

/**
 * \class L1GtParametersTrivialProducer
 *
 *
 * Description: ESProducer for L1 GT parameters.
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

#include <boost/cstdint.hpp>

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"

#include "CondFormats/L1TObjects/interface/L1GtParameters.h"
#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"

// forward declarations

// class declaration
class L1GtParametersTrivialProducer : public edm::ESProducer
{

public:

    /// constructor
    L1GtParametersTrivialProducer(const edm::ParameterSet&);

    /// destructor
    ~L1GtParametersTrivialProducer() override;


    /// public methods

    /// L1 GT parameters
    std::unique_ptr<L1GtParameters> produceGtParameters(
        const L1GtParametersRcd&);

private:

    /// total Bx's in the event
    int m_totalBxInEvent;

    /// active boards in the L1 DAQ record
    boost::uint16_t m_daqActiveBoards;

    /// active boards in the L1 EVM record
    boost::uint16_t m_evmActiveBoards;

    /// number of Bx per board in the DAQ record
    std::vector<int> m_daqNrBxBoard;

    /// number of Bx per board in the EVM record
    std::vector<int> m_evmNrBxBoard;

    /// length of BST record (in bytes)
    unsigned int m_bstLengthBytes;

};

#endif
