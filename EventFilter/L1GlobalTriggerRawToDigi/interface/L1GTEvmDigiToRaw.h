#ifndef EventFilter_L1GlobalTriggerRawToDigi_L1GTEvmDigiToRaw_h
#define EventFilter_L1GlobalTriggerRawToDigi_L1GTEvmDigiToRaw_h

/**
 * \class L1GTEvmDigiToRaw
 *
 *
 * Description: generate raw data from digis.
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

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/typedefs.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

// forward declarations
class FEDRawDataCollection;

class L1GtfeWord;
class L1GtfeExtWord;
class L1TcsWord;
class L1GtFdlWord;

// class declaration
class L1GTEvmDigiToRaw : public edm::stream::EDProducer<>
{

public:

    /// constructor(s)
    explicit L1GTEvmDigiToRaw(const edm::ParameterSet&);

private:

    /// loop over events
    void produce(edm::Event&, const edm::EventSetup&) override;

    /// block packers -------------

    /// pack header
    void packHeader(unsigned char*, edm::Event&);

    /// pack the GTFE block
    /// gives the number of bunch crosses in the event, as well as the active boards
    /// records for inactive boards are not written in the GT EVM record
    void packGTFE(const edm::EventSetup&, unsigned char*, L1GtfeExtWord&,
                  cms_uint16_t activeBoardsGtValue);

    /// pack the TCS block
    void packTCS(const edm::EventSetup& evSetup, unsigned char* ptrGt,
                 L1TcsWord& tcsBlock);

    /// pack FDL blocks for various bunch crosses
    void packFDL(const edm::EventSetup&, unsigned char*, L1GtFdlWord&);

    /// pack trailer word
    void packTrailer(unsigned char*, unsigned char*, int);

private:

    /// FED Id for GT EVM record
    /// default value defined in DataFormats/FEDRawData/src/FEDNumbering.cc
    int m_evmGtFedId;

    /// input tag for GT EVM record
    const edm::EDGetTokenT<L1GlobalTriggerEvmReadoutRecord> m_evmGtInputToken;
    const edm::InputTag m_evmGtInputTag;

    /// mask for active boards
    cms_uint16_t m_activeBoardsMaskGt;

    /// total Bx's in the event, obtained from GTFE block
    int m_totalBxInEvent;

    /// min Bx's in the event, computed after m_totalBxInEvent is obtained from GTFE block
    /// assume symmetrical number of BX around L1Accept
    int m_minBxInEvent;

    /// max Bx's in the event, computed after m_totalBxInEvent is obtained from GTFE block
    /// assume symmetrical number of BX around L1Accept
    int m_maxBxInEvent;

    /// length of BST record (in bytes)
    int m_bstLengthBytes;

private:

    /// verbosity level
    const int m_verbosity;
    const bool m_isDebugEnabled;

};

#endif // EventFilter_L1GlobalTriggerRawToDigi_L1GTEvmDigiToRaw_h
