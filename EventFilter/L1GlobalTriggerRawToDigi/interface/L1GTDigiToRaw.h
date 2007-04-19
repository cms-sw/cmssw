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
#include "FWCore/ParameterSet/interface/InputTag.h"

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
    virtual void beginJob(const edm::EventSetup&);

    /// loop over events
    virtual void produce(edm::Event&, const edm::EventSetup&);

    /// block packers -------------

    /// pack header
    void packHeader(const unsigned char*);

    /// pack the GTFE block
    /// gives the number of bunch crosses in the event, as well as the active boards
    /// records for inactive boards are not written in the GT DAQ record
    void packGTFE(const edm::EventSetup&, unsigned char*, L1GtfeWord&);

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

    /// pack trailer word
    void packTrailer(const unsigned char*);

    /// end of job stuff
    virtual void endJob();

private:

    /// input tags for GT DAQ record
    edm::InputTag m_daqGtInputTag;
    
    /// total Bx's in the event, obtained from GTFE block    
    int m_totalBxInEvent;
    
    /// mask for active boards (actually 16 bits)
    boost::uint16_t m_activeBoardsMaskGt;


};

#endif // EventFilter_L1GlobalTriggerRawToDigi_L1GTDigiToRaw_h
