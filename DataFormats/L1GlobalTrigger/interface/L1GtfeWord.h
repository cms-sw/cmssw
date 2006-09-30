#ifndef L1GlobalTrigger_L1GtfeWord_h
#define L1GlobalTrigger_L1GtfeWord_h

/**
 * \class L1GtfeWord
 * 
 * 
 * 
 * Description: L1 Global Trigger - GTFE words in the readout record 
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
//   base class

// forward declarations

class L1GtfeWord
{

public:

    static const int NumberGpsTimes = 8;

public:
    /// constructors
    L1GtfeWord();    // empty constructor, all members set to zero;

    // constructor from unpacked values;
    L1GtfeWord(
        uint16_t boardIdValue,
        uint16_t recordLengthValue,  
        uint16_t bxNrValue, 
        uint32_t setupVersionValue, 
        uint16_t activeBoardsValue,
        uint32_t totalTriggerNrValue,
        uint16_t gpsTimeValue[NumberGpsTimes]            
    ); 
    

    /// destructor
    virtual ~L1GtfeWord();

public:

    /// get/set board ID
    inline const uint16_t boardId() const { return m_boardId; }
    void setBoardId(uint16_t boardIdValue) { m_boardId = boardIdValue; }
     
    /// get/set record length: 3 bx for standard, 5 bx for debug
    inline const uint16_t recordLength() const { return m_recordLength; }
    void setRecordLength(uint16_t recordLengthValue) { m_recordLength = recordLengthValue; }
     
    /// get/set bunch cross number as counted in the GTFE board
    inline const uint16_t bxNr() const { return m_bxNr; }
    void setBxNr(uint16_t bxNrValue) { m_bxNr = bxNrValue; }
     
    /// get/set setup version
    inline const uint32_t setupVersion() const { return m_setupVersion; }
    void setSetupVersion(uint32_t setupVersionValue) { m_setupVersion = setupVersionValue; }
     
    /// get/set boards contributing to EVM respectively DAQ record
    inline const uint16_t activeBoards() const { return m_activeBoards; }
    void setActiveBoards(uint16_t activeBoardsValue) { m_activeBoards = activeBoardsValue; }
     
    /// get/set total number of L1A sent since start of run
    inline const uint32_t totalTriggerNr() const { return m_totalTriggerNr; }
    void setTotalTriggerNr(uint32_t totalTriggerNrValue) { m_totalTriggerNr = totalTriggerNrValue; }

    /// get/set GPS time for index iB
    const uint16_t gpsTime(int iB) const;
    void setGpsTime(uint16_t gpsTimeVal, int iB);
        
private:

                               // first number in the comment represents number of bits

    uint16_t m_boardId;        //  8: board identifier

    uint16_t m_recordLength;   //  4: record length: 3 bx for standard, 5 bx for debug  

    uint16_t m_bxNr;           // 12: bunch cross number as counted in the GTFE board 
    uint32_t m_setupVersion;   // 32: 
//
    uint16_t m_activeBoards;   // 16: boards contributing to EVM respectively DAQ record

    uint32_t m_totalTriggerNr; // 32: total number of L1A sent since start of run
//
    uint16_t m_gpsTime[NumberGpsTimes];     //  8:            
        
};

#endif /*L1GlobalTrigger_L1GtfeWord_h*/
