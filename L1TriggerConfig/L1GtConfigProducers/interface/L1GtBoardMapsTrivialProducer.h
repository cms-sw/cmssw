#ifndef L1GtConfigProducers_L1GtBoardMapsTrivialProducer_h
#define L1GtConfigProducers_L1GtBoardMapsTrivialProducer_h

/**
 * \class L1GtBoardMapsTrivialProducer
 *
 *
 * Description: ESProducer for mappings of the L1 GT boards.
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

#include "CondFormats/L1TObjects/interface/L1GtBoard.h"
#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"
#include "CondFormats/DataRecord/interface/L1GtBoardMapsRcd.h"

// forward declarations

// class declaration
class L1GtBoardMapsTrivialProducer : public edm::ESProducer
{

public:

    /// constructor
    L1GtBoardMapsTrivialProducer(const edm::ParameterSet&);

    /// destructor
    ~L1GtBoardMapsTrivialProducer() override;


    /// public methods

    /// produce mappings of the L1 GT boards
    std::unique_ptr<L1GtBoardMaps> produceBoardMaps(
        const L1GtBoardMapsRcd&);

private:

    /// L1 GT boards and their mapping
    std::vector<L1GtBoard> m_gtBoardMaps;

};

#endif
