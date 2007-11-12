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

class L1GtfeWord;
class L1GtfeExtWord;
class L1TcsWord;
class L1GtFdlWord;

// class declaration
class L1GTEvmDigiToRaw : public edm::EDProducer
{

public:

    /// constructor(s)
    explicit L1GTEvmDigiToRaw(const edm::ParameterSet&);

    /// destructor
    virtual ~L1GTEvmDigiToRaw();

private:

    /// beginning of job stuff
    virtual void beginJob(const edm::EventSetup&);

    /// loop over events
    virtual void produce(edm::Event&, const edm::EventSetup&);

    /// block packers -------------

    /// pack header
    void packHeader(unsigned char*);

    /// pack the GTFE block
    /// gives the number of bunch crosses in the event, as well as the active boards
    /// records for inactive boards are not written in the GT DAQ record
    void packGTFE(const edm::EventSetup&, unsigned char*, L1GtfeExtWord&,
                  boost::uint16_t activeBoardsGtValue);

    /// pack the TCS block
    void packTCS(const edm::EventSetup& evSetup, unsigned char* ptrGt,
                 L1TcsWord& tcsBlock);

    /// pack FDL blocks for various bunch crosses
    void packFDL(const edm::EventSetup&, unsigned char*, L1GtFdlWord&);

    /// pack trailer word
    void packTrailer(unsigned char*, int);

    /// end of job stuff
    virtual void endJob();

private:

    /// input tag for GT DAQ record
    edm::InputTag m_evmGtInputTag;

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


};

#endif // EventFilter_L1GlobalTriggerRawToDigi_L1GTEvmDigiToRaw_h
