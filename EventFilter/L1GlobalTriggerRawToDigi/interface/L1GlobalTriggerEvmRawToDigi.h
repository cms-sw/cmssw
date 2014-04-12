#ifndef EventFilter_L1GlobalTriggerRawToDigi_L1GlobalTriggerEvmRawToDigi_h
#define EventFilter_L1GlobalTriggerRawToDigi_L1GlobalTriggerEvmRawToDigi_h

/**
 * \class L1GlobalTriggerEvmRawToDigi
 *
 *
 * Description: unpack EVM raw data into digitized data.
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
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/typedefs.h"

// forward declarations
class L1GtfeWord;
class L1GtfeExtWord;
class L1TcsWord;
class L1GtFdlWord;

class FEDHeader;
class FEDTrailer;


// class declaration
class L1GlobalTriggerEvmRawToDigi : public edm::EDProducer
{

public:

    /// constructor(s)
    explicit L1GlobalTriggerEvmRawToDigi(const edm::ParameterSet&);

    /// destructor
    virtual ~L1GlobalTriggerEvmRawToDigi();

private:

    virtual void produce(edm::Event&, const edm::EventSetup&) override;

    /// block unpackers

    /// unpack header
    void unpackHeader(const unsigned char*, FEDHeader&);

    /// unpack trailer word
    void unpackTrailer(const unsigned char*, FEDTrailer&);

    /// produce empty products in case of problems
    void produceEmptyProducts(edm::Event&);

    /// dump FED raw data
    void dumpFedRawData(const unsigned char*, int, std::ostream&);

private:

    L1GtfeExtWord* m_gtfeWord;
    L1TcsWord* m_tcsWord;
    L1GtFdlWord* m_gtFdlWord;

    /// input tags for GT EVM record
    edm::InputTag m_evmGtInputTag;

    /// FED Id for GT EVM record
    /// default value defined in DataFormats/FEDRawData/src/FEDNumbering.cc
    int m_evmGtFedId;

    /// mask for active boards
    cms_uint16_t m_activeBoardsMaskGt;

    // number of bunch crossing to be unpacked
    int m_unpackBxInEvent;

    /// lowest bxInEvent to be unpacked in the event
    /// assume symmetrical number of BX around L1Accept
    int m_lowSkipBxInEvent;

    /// upper bxInEvent to be unpacked in the event
    /// assume symmetrical number of BX around L1Accept
    int m_uppSkipBxInEvent;

    /// total Bx's in the event, obtained from GTFE block
    //
    /// corresponding to alternative 0 in altNrBxBoard()
    int m_recordLength0;

    /// corresponding to alternative 1 in altNrBxBoard()
    int m_recordLength1;

    /// number of Bx for a board, obtained from GTFE block (record length & alternative)
    int m_totalBxInEvent;


    /// length of BST record (in bytes)
    int m_bstLengthBytes;

private:

    /// verbosity level
    int m_verbosity;
    bool m_isDebugEnabled;



};

#endif // EventFilter_L1GlobalTriggerRawToDigi_L1GlobalTriggerEvmRawToDigi_h
