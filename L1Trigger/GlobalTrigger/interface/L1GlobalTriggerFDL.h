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
#include <bitset>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

#include "FWCore/Framework/interface/EventSetup.h"

// forward declarations
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
    void run(int iBxInEvent, const edm::EventSetup& evSetup);

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
