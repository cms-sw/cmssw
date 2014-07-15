#ifndef EventFilter_L1GlobalTriggerRawToDigi_L1GlobalTriggerRawToDigi_h
#define EventFilter_L1GlobalTriggerRawToDigi_L1GlobalTriggerRawToDigi_h

/**
 * \class L1GlobalTriggerRawToDigi
 *
 *
 * Description: unpack raw data into digitized data.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna -  GT
 * \author: Ivan Mikulec       - HEPHY Vienna - GMT
 *
 *
 */

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/typedefs.h"

// forward declarations
class L1GtfeWord;
class L1GtFdlWord;
class L1GtPsbWord;

class L1MuGMTReadoutCollection;

class FEDHeader;
class FEDTrailer;

class L1MuTriggerScales;
class L1MuTriggerPtScale;


// class declaration
class L1GlobalTriggerRawToDigi : public edm::stream::EDProducer<>
{

public:

    /// constructor(s)
    explicit L1GlobalTriggerRawToDigi(const edm::ParameterSet&);

    /// destructor
    virtual ~L1GlobalTriggerRawToDigi();

private:

    virtual void produce(edm::Event&, const edm::EventSetup&) override;

    /// block unpackers

    /// unpack header
    void unpackHeader(const unsigned char*, FEDHeader&);

    /// unpack PSB blocks
    /// unpacking is done in PSB class format
    /// methods are given later to translate from the PSB format
    /// to the physical input of the PSB
    void unpackPSB(const edm::EventSetup&, const unsigned char*, L1GtPsbWord&);

    /// unpack the GMT record
    void unpackGMT(const unsigned char*, std::auto_ptr<L1MuGMTReadoutCollection>&,edm::Event&);

    /// unpack trailer word
    void unpackTrailer(const unsigned char*, FEDTrailer&);


    /// produce empty products in case of problems
    void produceEmptyProducts(edm::Event&);


    /// dump FED raw data
    void dumpFedRawData(const unsigned char*, int, std::ostream&);


private:

    L1GtfeWord* m_gtfeWord;
    L1GtPsbWord* m_gtPsbWord;
    L1GtFdlWord* m_gtFdlWord;

    /// input tags for GT DAQ record
    edm::InputTag m_daqGtInputTag;

    /// FED Id for GT DAQ record
    /// default value defined in DataFormats/FEDRawData/src/FEDNumbering.cc
    int m_daqGtFedId;

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


    /// muon trigger scales to convert unpacked data into physical quantities
    const L1MuTriggerScales* m_TriggerScales;
    const L1MuTriggerPtScale* m_TriggerPtScale;

private:

    /// verbosity level
    int m_verbosity;
    bool m_isDebugEnabled;


};

#endif // EventFilter_L1GlobalTriggerRawToDigi_L1GlobalTriggerRawToDigi_h
