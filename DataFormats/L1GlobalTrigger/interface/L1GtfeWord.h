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
#include <iosfwd>

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
        boost::uint16_t recordLength1Value,
        boost::uint16_t recordLengthValue,
        boost::uint16_t bxNrValue,
        boost::uint32_t setupVersionValue,
        boost::uint16_t activeBoardsValue,
        boost::uint16_t altNrBxBoardValue,
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

    /// set the BoardId value from a 64-bits word
    void setBoardId(const boost::uint64_t& word64);

    /// set the BoardId value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setBoardIdWord64(boost::uint64_t& word64, int iWord);


    /// get/set record length for alternative 1
    inline const boost::uint16_t recordLength1() const
    {
        return m_recordLength1;
    }

    void setRecordLength1(boost::uint16_t recordLengthValue)
    {
        m_recordLength1 = recordLengthValue;
    }

    void setRecordLength1(const boost::uint64_t& word64);

    /// set the RecordLength value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setRecordLength1Word64(boost::uint64_t& word64, int iWord);



    /// get/set record length for alternative 0
    inline const boost::uint16_t recordLength() const
    {
        return m_recordLength;
    }

    void setRecordLength(boost::uint16_t recordLengthValue)
    {
        m_recordLength = recordLengthValue;
    }

    void setRecordLength(const boost::uint64_t& word64);

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

    void setBxNr(const boost::uint64_t& word64);

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

    void setSetupVersion(const boost::uint64_t& word64);

    /// set the SetupVersion value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setSetupVersionWord64(boost::uint64_t& word64, int iWord);


    /// get / set BST flag: 0 or 1 - via setup version (no private member)
    const int bstFlag() const;
    void setBstFlag(const int);


    /// get/set boards contributing to EVM respectively DAQ record
    inline const boost::uint16_t activeBoards() const
    {
        return m_activeBoards;
    }

    void setActiveBoards(boost::uint16_t activeBoardsValue)
    {
        m_activeBoards = activeBoardsValue;
    }

    void setActiveBoards(const boost::uint64_t& word64);

    /// set the ActiveBoards value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setActiveBoardsWord64(boost::uint64_t& word64, int iWord);

    /// set the ActiveBoards value in a 64-bits word, having the index iWord
    /// in the GTFE raw record from the value activeBoardsValue
    void setActiveBoardsWord64(boost::uint64_t& word64, int iWord,
                               boost::int16_t activeBoardsValue);



    /// get/set alternative for number of BX per board
    inline const boost::uint16_t altNrBxBoard() const
    {
        return m_altNrBxBoard;
    }

    void setAltNrBxBoard(boost::uint16_t altNrBxBoardValue)
    {
        m_altNrBxBoard = altNrBxBoardValue;
    }

    void setAltNrBxBoard(const boost::uint64_t& word64);

    /// set the AltNrBxBoard value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setAltNrBxBoardWord64(boost::uint64_t& word64, int iWord);

    /// set the AltNrBxBoard value in a 64-bits word, having the index iWord
    /// in the GTFE raw record from the value altNrBxBoardValue
    void setAltNrBxBoardWord64(boost::uint64_t& word64, int iWord,
                               boost::int16_t altNrBxBoardValue);



    /// get/set total number of L1A sent since start of run
    inline const boost::uint32_t totalTriggerNr() const
    {
        return m_totalTriggerNr;
    }

    void setTotalTriggerNr(boost::uint32_t totalTriggerNrValue)
    {
        m_totalTriggerNr = totalTriggerNrValue;
    }

    void setTotalTriggerNr(const boost::uint64_t& word64);

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
    virtual void reset();

    /// pretty print the content of a L1GtfeWord
    virtual void print(std::ostream& myCout) const;

    /// unpack GTFE
    /// gtfePtr pointer to the beginning of the GTFE block in the raw data
    virtual void unpack(const unsigned char* gtfePtr);

private:

    // block description in the raw GT record

    // block size in 64bits words
    static const int BlockSize;

    static const int BoardIdWord;
    static const int RecordLength1Word;
    static const int RecordLengthWord;
    static const int BxNrWord;
    static const int SetupVersionWord;

    static const boost::uint64_t BoardIdMask;
    static const boost::uint64_t RecordLength1Mask;
    static const boost::uint64_t RecordLengthMask;
    static const boost::uint64_t BxNrMask;
    static const boost::uint64_t SetupVersionMask;

    static const boost::uint32_t BstFlagMask;

    // shifts could be computed from masks...
    static const int BoardIdShift;
    static const int RecordLength1Shift;
    static const int RecordLengthShift;
    static const int BxNrShift;
    static const int SetupVersionShift;

    //
    static const int ActiveBoardsWord;
    static const int AltNrBxBoardWord;
    static const int TotalTriggerNrWord;

    static const boost::uint64_t ActiveBoardsMask;
    static const boost::uint64_t AltNrBxBoardMask;
    static const boost::uint64_t TotalTriggerNrMask;

    static const int ActiveBoardsShift;
    static const int AltNrBxBoardShift;
    static const int TotalTriggerNrShift;


private:

    /// board identifier
    boost::uint16_t m_boardId;

    /// record length for alternative 1
    boost::uint16_t m_recordLength1;

    /// record length  for alternative 0
    boost::uint16_t m_recordLength;

    /// bunch cross number as counted in the GTFE board
    boost::uint16_t m_bxNr;
    boost::uint32_t m_setupVersion;

    /// active boards contributing to EVM respectively DAQ record
    boost::uint16_t m_activeBoards;

    /// alternative for number of BX per board
    ///     correlated with active boards
    ///     bit value is 0: take alternative 0
    ///     bit value is 1: take alternative 1
    boost::uint16_t m_altNrBxBoard;

    /// total number of L1A sent since start of run
    boost::uint32_t m_totalTriggerNr;

};

#endif /*L1GlobalTrigger_L1GtfeWord_h*/
