#ifndef GlobalTrigger_L1GlobalTriggerFDL_h
#define GlobalTrigger_L1GlobalTriggerFDL_h
/**
 * \class L1GlobalTriggerFDL
 * 
 * 
 * Description: Final Decision Logic board.
 *  
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: M. Fierro            - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <vector>

#include <boost/cstdint.hpp>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtBoard.h"

// forward declarations
class L1GlobalTriggerReadoutRecord;
class L1GlobalTriggerEvmReadoutRecord;

class L1GlobalTrigger;
class L1GtFdlWord;

// class declaration
class L1GlobalTriggerFDL
{

public:

    /// constructor
    L1GlobalTriggerFDL(L1GlobalTrigger& gt);

    /// destructor
    virtual ~L1GlobalTriggerFDL();

    /// run the FDL
    void run(
        edm::Event& iEvent,
        const edm::EventSetup& evSetup,
        const std::vector<L1GtBoard>& boardMaps,
        const int totalBxInEvent,
        const int iBxInEvent);

    /// fill the FDL block in the L1 GT DAQ record for iBxInEvent
    void fillDaqFdlBlock(
        const boost::uint16_t& activeBoardsGtDaq,
        const std::vector<L1GtBoard>& boardMaps,
        std::auto_ptr<L1GlobalTriggerReadoutRecord>& gtDaqReadoutRecord);

    /// fill the FDL block in the L1 GT EVM record for iBxInEvent
    void fillEvmFdlBlock(
        const boost::uint16_t& activeBoardsGtEvm,
        const std::vector<L1GtBoard>& boardMaps,
        std::auto_ptr<L1GlobalTriggerEvmReadoutRecord>& gtEvmReadoutRecord);

    /// clear FDL
    void reset();

    /// return the GtFdlWord
    inline L1GtFdlWord* gtFdlWord() const
    {
        return m_gtFdlWord;
    }



private:

    const L1GlobalTrigger& m_GT;

    L1GtFdlWord* m_gtFdlWord;

    /// prescale counters: NumberPhysTriggers counters per bunch cross in event
    std::vector<std::vector<int> > m_prescaleCounter;

    /// logical switches for
    ///    the first event
    ///    the first event in the luminosity segment
    ///    and the first event in the run
    bool m_firstEv;
    bool m_firstEvLumiSegment;
    bool m_firstEvRun;

};

#endif
