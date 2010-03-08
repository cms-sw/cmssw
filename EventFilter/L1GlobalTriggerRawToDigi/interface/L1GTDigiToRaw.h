#ifndef EventFilter_L1GlobalTriggerRawToDigi_L1GTDigiToRaw_h
#define EventFilter_L1GlobalTriggerRawToDigi_L1GTDigiToRaw_h

/**
 * \class L1GTDigiToRaw
 *
 *
 * Description: generate raw data from digis.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna -  GT
 * \author: Ivan Mikulec       - HEPHY Vienna - GMT
 *
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <memory>

#include <boost/cstdint.hpp>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

// forward declarations
class FEDRawDataCollection;
class L1MuGMTReadoutRecord;
class L1MuGMTReadoutCollection;

class L1GtfeWord;
class L1GtFdlWord;
class L1GtPsbWord;

// class declaration
class L1GTDigiToRaw : public edm::EDProducer
{

public:

    /// constructor(s)
    explicit L1GTDigiToRaw(const edm::ParameterSet&);

    /// destructor
    virtual ~L1GTDigiToRaw();

private:

    /// beginning of job stuff
    virtual void beginJob();

    /// loop over events
    virtual void produce(edm::Event&, const edm::EventSetup&);

    /// block packers -------------

    /// pack header
    void packHeader(unsigned char*, edm::Event&);

    /// pack the GTFE block
    /// gives the number of bunch crosses in the event, as well as the active boards
    /// records for inactive boards are not written in the GT DAQ record
    void packGTFE(const edm::EventSetup&, unsigned char*, L1GtfeWord&,
                  boost::uint16_t activeBoardsGtValue);

    /// pack FDL blocks for various bunch crosses
    void packFDL(const edm::EventSetup&, unsigned char*, L1GtFdlWord&);

    /// pack PSB blocks
    /// packing is done in PSB class format
    void packPSB(const edm::EventSetup&, unsigned char*, L1GtPsbWord&);

    /// pack the GMT collection using packGMT (GMT record packing)
    unsigned int packGmtCollection(
        unsigned char* ptrGt,
        L1MuGMTReadoutCollection const* digis);

    /// pack a GMT record
    unsigned int packGMT(L1MuGMTReadoutRecord const&, unsigned char*);
    unsigned int flipPtQ(unsigned int);

    /// pack trailer word
    void packTrailer(unsigned char*, unsigned char*, int);

    /// end of job stuff
    virtual void endJob();

private:

    /// FED Id for GT DAQ record
    /// default value defined in DataFormats/FEDRawData/src/FEDNumbering.cc
    int m_daqGtFedId;

    /// input tag for GT DAQ record
    edm::InputTag m_daqGtInputTag;

    /// input tag for GMT record
    edm::InputTag m_muGmtInputTag;

    /// mask for active boards
    boost::uint16_t m_activeBoardsMaskGt;

    /// total Bx's in the event, obtained from GTFE block
    int m_totalBxInEvent;

    /// min Bx's in the event, computed after m_totalBxInEvent is obtained from GTFE block
    /// assume symmetrical number of BX around L1Accept
    int m_minBxInEvent;

    /// max Bx's in the event, computed after m_totalBxInEvent is obtained from GTFE block
    /// assume symmetrical number of BX around L1Accept
    int m_maxBxInEvent;

private:

    /// verbosity level
    int m_verbosity;
    bool m_isDebugEnabled;

};

#endif // EventFilter_L1GlobalTriggerRawToDigi_L1GTDigiToRaw_h
