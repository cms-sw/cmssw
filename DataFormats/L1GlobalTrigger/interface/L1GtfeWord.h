#ifndef L1GlobalTrigger_L1GtfeWord_h
#define L1GlobalTrigger_L1GtfeWord_h

/**
 * \class L1GtfeWord
 * 
 * 
 * Description: L1 Global Trigger - GTFE words in the readout record.  
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
#include <boost/cstdint.hpp>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

// forward declarations

// class interface

class L1GtfeWord
{

public:

    /// constructors
    L1GtfeWord();    // empty constructor, all members set to zero;

    /// constructor from unpacked values;
    L1GtfeWord(
        boost::uint16_t boardIdValue,
        boost::uint16_t recordLengthValue,
        boost::uint16_t bxNrValue,
        boost::uint32_t setupVersionValue,
        boost::uint16_t activeBoardsValue,
        boost::uint32_t totalTriggerNrValue
    );


    /// destructor
    virtual ~L1GtfeWord();

    /// equal operator
    bool operator==(const L1GtfeWord&) const;

    /// unequal operator
    bool operator!=(const L1GtfeWord&) const;

public:

    /// get/set board ID

    /// get BoardId value
    inline const boost::uint16_t boardId() const
    {
        return m_boardId;
    }

    /// set BoardId from a BoardId value
    void setBoardId(boost::uint16_t boardIdValue)
    {
        m_boardId = boardIdValue;
    }

    /// set the BoardId value from a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setBoardId(const boost::uint64_t& word64, int iWord);

    /// set the BoardId value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setBoardIdWord64(boost::uint64_t& word64, int iWord);


    /// get/set record length: 3 bx for standard, 5 bx for debug
    inline const boost::uint16_t recordLength() const
    {
        return m_recordLength;
    }

    void setRecordLength(boost::uint16_t recordLengthValue)
    {
        m_recordLength = recordLengthValue;
    }

    void setRecordLength(const boost::uint64_t& word64, int iWord);

    /// set the RecordLength value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setRecordLengthWord64(boost::uint64_t& word64, int iWord);


    /// get/set bunch cross number as counted in the GTFE board
    inline const boost::uint16_t bxNr() const
    {
        return m_bxNr;
    }

    void setBxNr(boost::uint16_t bxNrValue)
    {
        m_bxNr = bxNrValue;
    }

    void setBxNr(const boost::uint64_t& word64, int iWord);

    /// set the BxNr value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setBxNrWord64(boost::uint64_t& word64, int iWord);

    /// get/set setup version
    inline const boost::uint32_t setupVersion() const
    {
        return m_setupVersion;
    }

    void setSetupVersion(boost::uint32_t setupVersionValue)
    {
        m_setupVersion = setupVersionValue;
    }

    void setSetupVersion(const boost::uint64_t& word64, int iWord);

    /// set the SetupVersion value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setSetupVersionWord64(boost::uint64_t& word64, int iWord);

    /// get/set boards contributing to EVM respectively DAQ record
    inline const boost::uint16_t activeBoards() const
    {
        return m_activeBoards;
    }

    void setActiveBoards(boost::uint16_t activeBoardsValue)
    {
        m_activeBoards = activeBoardsValue;
    }

    void setActiveBoards(const boost::uint64_t& word64, int iWord);

    /// set the ActiveBoards value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setActiveBoardsWord64(boost::uint64_t& word64, int iWord);

    /// set the ActiveBoards value in a 64-bits word, having the index iWord
    /// in the GTFE raw record from the value activeBoardsValue
    void setActiveBoardsWord64(boost::uint64_t& word64, int iWord,
                               boost::int16_t activeBoardsValue);

    /// get/set total number of L1A sent since start of run
    inline const boost::uint32_t totalTriggerNr() const
    {
        return m_totalTriggerNr;
    }

    void setTotalTriggerNr(boost::uint32_t totalTriggerNrValue)
    {
        m_totalTriggerNr = totalTriggerNrValue;
    }

    void setTotalTriggerNr(const boost::uint64_t& word64, int iWord);

    /// set the TotalTriggerNr value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setTotalTriggerNrWord64(boost::uint64_t& word64, int iWord);


    /// get the size of the GTFE block in GT DAQ record (in multiple of 8 bits)
    inline const unsigned int getSize() const
    {
        int unitLengthBits = L1GlobalTriggerReadoutSetup::UnitLength;

        return BlockSize*unitLengthBits;
    }

public:

    /// reset the content of a L1GtfeWord
    void reset();

    /// pretty print the content of a L1GtfeWord
    void print(std::ostream& myCout) const;

private:

    // block description in the raw GT record

    // block size in 64bits words
    static const int BlockSize = 2;        // 2 x 64bits

    static const int BoardIdWord = 0;
    static const int RecordLengthWord = 0;
    static const int BxNrWord = 0;
    static const int SetupVersionWord = 0;

    static const boost::uint64_t BoardIdMask =      0xFF00000000000000ULL;
    static const boost::uint64_t RecordLengthMask = 0x001F000000000000ULL;
    static const boost::uint64_t BxNrMask =         0x00000FFF00000000ULL;
    static const boost::uint64_t SetupVersionMask = 0x00000000FFFFFFFFULL;

    // shifts could be computed from masks...
    static const int BoardIdShift = 56;
    static const int RecordLengthShift = 48;
    static const int BxNrShift = 32;
    static const int SetupVersionShift = 0;

    //
    static const int ActiveBoardsWord = 1;
    static const int TotalTriggerNrWord = 1;

    static const boost::uint64_t ActiveBoardsMask =   0xFFFF000000000000ULL;
    static const boost::uint64_t TotalTriggerNrMask = 0x00000000FFFFFFFFULL;

    static const int ActiveBoardsShift = 48;
    static const int TotalTriggerNrShift = 0;


private:

    // first number in the comment represents number of bits

    boost::uint16_t m_boardId;        //  8: board identifier

    boost::uint16_t m_recordLength;   //  5: record length: 3 bx for standard, 5 bx for debug

    boost::uint16_t m_bxNr;           // 12: bunch cross number as counted in the GTFE board
    boost::uint32_t m_setupVersion;   // 32:
    //
    boost::uint16_t m_activeBoards;   // 16: boards contributing to EVM respectively DAQ record

    boost::uint32_t m_totalTriggerNr; // 32: total number of L1A sent since start of run

};

#endif /*L1GlobalTrigger_L1GtfeWord_h*/
