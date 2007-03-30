#ifndef L1GlobalTrigger_L1GtPsbWord_h
#define L1GlobalTrigger_L1GtPsbWord_h

/**
 * \class L1GtPsbWord
 * 
 * 
 * Description: PSB block in the L1 GT readout record.  
 *
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
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

// forward declarations

// class declaration
class L1GtPsbWord
{

public:

    static const int NumberAData = 8;
    static const int NumberBData = 8;

public:
    /// constructors
    L1GtPsbWord();    // empty constructor, all members set to zero;

    /// constructor from unpacked values;
    L1GtPsbWord(
        uint16_t boardIdValue,
        uint16_t bxInEventValue,
        uint16_t bxNrValue,
        uint32_t eventNrValue,
        uint16_t aDataValue[NumberAData],
        uint16_t bDataValue[NumberBData],
        uint16_t localBxNrValue
    ); 
    

    /// destructor
    virtual ~L1GtPsbWord();

    /// equal operator
    bool operator==(const L1GtPsbWord&) const;

    /// unequal operator
    bool operator!=(const L1GtPsbWord&) const;


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

    /// get/set event number since last L1 reset generated in PSB
    inline const uint32_t eventNr() const { return m_eventNr; } 
    void setEventNr(uint32_t eventNrValue) { m_eventNr = eventNrValue; }

    /// get/set A_DATA_CH_IA 
    const uint16_t aData(int iA) const;
    void setAData(uint16_t aDataVal, int iA);

    /// get/set B_DATA_CH_IB 
    const uint16_t bData(int iB) const;
    void setBData(uint16_t bDataVal, int iB);
    
    /// get/set local bunch cross number of the actual bx
    inline const uint16_t localBxNr() const { return m_localBxNr; }     
    void setLocalBxNr(uint16_t localBxNrValue) { m_localBxNr = localBxNrValue; }

    /// get the size of the PSB block in GT DAQ record (in multiple of 8 bits)
    inline const unsigned int getSize() const
    {
        int unitLengthBits = L1GlobalTriggerReadoutSetup::UnitLength;

        return BlockSize*unitLengthBits;
    }
    

public:

    /// reset the content of a L1GtPsbWord
    void reset();

private:

    // block description in the raw GT record

    // block size in 64bits words (BlockSize * 64 bits)
    static const int BlockSize = 6;        
                            
private:

                               // first number in the comment represents number of bits

    uint16_t m_boardId;        // 16: board identifier
                               //
    uint16_t m_bxInEvent;      //  4: bunch cross in the GT event record 
                               //     one keeps 3 bx (F, 0, 1) for standard record, 
                               //               5 bx (E, F, 0, 1) for debug record  
                               //
    uint16_t m_bxNr;           // 12: bunch cross number of the actual bx
    uint32_t m_eventNr;        // 24: event number since last L1 reset generated in PSB
//
    uint16_t m_aData[NumberAData];    // 16: A_Data_ChX
//
    uint16_t m_bData[NumberBData];    // 16: B_Data_ChX
//
    uint16_t m_localBxNr;      // 12: local bunch cross number of the actual bx
                               //     bx number at which the data were written into the ringbuffer

        
};

#endif /*L1GlobalTrigger_L1GtPsbWord_h*/
