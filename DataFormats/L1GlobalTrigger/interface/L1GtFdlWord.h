#ifndef L1GlobalTrigger_L1GtFdlWord_h
#define L1GlobalTrigger_L1GtFdlWord_h

/**
 * \class L1GtFdlWord
 * 
 * 
 * 
 * Description: L1 Global Trigger - FDL composed word in the readout record 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date:$
 * $Revision:$
 *
 */

// system include files
#include <boost/cstdint.hpp>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

// forward declarations

class L1GtFdlWord
{

public:
    /// constructors
    L1GtFdlWord();    // empty constructor, all members set to zero;

    // constructor from unpacked values;
    L1GtFdlWord(
        uint16_t boardIdValue,
        uint16_t bxInEventValue,
        uint16_t bxNrValue,
        uint32_t eventNrValue,
        L1GlobalTriggerReadoutRecord::TechnicalTriggerWord gtTechnicalTriggerValue,
        L1GlobalTriggerReadoutRecord::DecisionWord gtDecisionValue,
        L1GlobalTriggerReadoutRecord::DecisionWordExtended gtDecisionExtendedValue,
        uint16_t finalORValue,
        uint16_t localBxNrValue
    ); 
    

    /// destructor
    virtual ~L1GtFdlWord();

public:

    /// get/set board ID
    inline const uint16_t boardId() const { return m_boardId; }
    void setBoardId(uint16_t boardIdValue) { m_boardId = boardIdValue; }
          
    /// get/set bunch cross in the GT event record 
    inline const uint16_t bxInEvent() const { return m_bxInEvent; }
    void setBxInEvent(uint16_t bxInEventValue) { m_bxInEvent = bxInEventValue; }

    /// get/set bunch cross number of the actual bx
    inline const uint16_t bxNr() const { return m_bxNr; }
    void setBxNr(uint16_t bxNrValue) { m_bxNr = bxNrValue; }

    /// get/set event number since last L1 reset generated in FDL
    inline const uint32_t eventNr() const { return m_eventNr; } 
    void setEventNr(uint32_t eventNrValue) { m_eventNr = eventNrValue; }
    
    /// get/set technical trigger bits
    inline const L1GlobalTriggerReadoutRecord::TechnicalTriggerWord gtTechnicalTrigger() const { return m_gtTechnicalTrigger; }
    void setGtTechnicalTrigger (L1GlobalTriggerReadoutRecord::TechnicalTriggerWord gtTechnicalTriggerValue) { m_gtTechnicalTrigger = gtTechnicalTriggerValue; } 

    /// get/set algorithms bits (decision word) 
    inline const L1GlobalTriggerReadoutRecord::DecisionWord gtDecision() const { return m_gtDecision; }
    void setGtDecision (L1GlobalTriggerReadoutRecord::DecisionWord gtDecisionValue) { m_gtDecision = gtDecisionValue; } 

    /// get/set extended algorithms bits (extended decision word) 
    inline const L1GlobalTriggerReadoutRecord::DecisionWordExtended gtDecisionExtended() const { return m_gtDecisionExtended; }
    void setGtDecisionExtended (L1GlobalTriggerReadoutRecord::DecisionWordExtended gtDecisionExtendedValue) { m_gtDecisionExtended = gtDecisionExtendedValue; } 

    /// get/set "Final OR" bits
    inline const uint16_t finalOR() const { return m_finalOR; } 
    void setFinalOR(uint16_t finalORValue) { m_finalOR = finalORValue; }

    /// get/set local bunch cross number of the actual bx
    inline const uint16_t localBxNr() const { return m_localBxNr; }     
    void setLocalBxNr(uint16_t localBxNrValue) { m_localBxNr = localBxNrValue; }
            
private:

                               // first number in the comment represents number of bits

    uint16_t m_boardId;        //  8: board identifier
                               //
    uint16_t m_bxInEvent;      //  4: bunch cross in the GT event record 
                               //     one keeps 3 bx (F, 0, 1) for standard record, 
                               //               5 bx (E, F, 0, 1) for debug record  
                               //
    uint16_t m_bxNr;           // 12: bunch cross number of the actual bx
    uint32_t m_eventNr;        // 24: event number since last L1 reset generated in FDL
//
    L1GlobalTriggerReadoutRecord::TechnicalTriggerWord m_gtTechnicalTrigger;
                               // 64: technical trigger bits
                               
    L1GlobalTriggerReadoutRecord::DecisionWord m_gtDecision;
                               // 128: algorithm bits

    L1GlobalTriggerReadoutRecord::DecisionWordExtended m_gtDecisionExtended;
                               // 192: algorithm bits
//
    uint16_t m_finalOR;        // 16: FINOR (7:0) Final OR bits. 
                               //     FINOR(i) is connected to Partition (i); 
                               //     FINOR (15:8) not used
//
    uint16_t m_localBxNr;      // 12: local bunch cross number of the actual bx
                               //     bx number at which the data were written into the ringbuffer

        
};

#endif /*L1GlobalTrigger_L1GtFdlWord_h*/
