#ifndef L1GlobalTrigger_L1GtFdlWord_h
#define L1GlobalTrigger_L1GtFdlWord_h

/**
 * \class L1GtFdlWord
 * 
 * 
 * Description: L1 Global Trigger - FDL block in the readout record.  
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

class L1GtFdlWord
{

public:
    /// constructors
    L1GtFdlWord();    // empty constructor, all members set to zero;

    // constructor from unpacked values;
    L1GtFdlWord(
        boost::uint16_t boardIdValue,
        int bxInEventValue,
        boost::uint16_t bxNrValue,
        boost::uint32_t eventNrValue,
        TechnicalTriggerWord gtTechnicalTriggerWordValue,
        DecisionWord gtDecisionWordValue,
        DecisionWordExtended gtDecisionWordExtendedValue,
        boost::uint16_t noAlgoValue,
        boost::uint16_t finalORValue,
        boost::uint16_t localBxNrValue
    );


    /// destructor
    virtual ~L1GtFdlWord();

    /// equal operator
    bool operator==(const L1GtFdlWord&) const;

    /// unequal operator
    bool operator!=(const L1GtFdlWord&) const;

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



    /// get/set bunch cross in the GT event record
    inline const int bxInEvent() const
    {
        return m_bxInEvent;
    }

    void setBxInEvent(int bxInEventValue)
    {
        m_bxInEvent = bxInEventValue;
    }

    /// set the BxInEvent value from a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setBxInEvent(const boost::uint64_t& word64, int iWord);

    /// set the BxInEvent value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setBxInEventWord64(boost::uint64_t& word64, int iWord);



    /// get/set BxNr - bunch cross number of the actual bx
    inline const boost::uint16_t bxNr() const
    {
        return m_bxNr;
    }

    void setBxNr(boost::uint16_t bxNrValue)
    {
        m_bxNr = bxNrValue;
    }

    /// set the BxNr value from a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setBxNr(const boost::uint64_t& word64, int iWord);

    /// set the BxNr value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setBxNrWord64(boost::uint64_t& word64, int iWord);



    /// get/set event number since last L1 reset generated in FDL
    inline const boost::uint32_t eventNr() const
    {
        return m_eventNr;
    }

    void setEventNr(boost::uint32_t eventNrValue)
    {
        m_eventNr = eventNrValue;
    }

    /// set the EventNr value from a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setEventNr(const boost::uint64_t& word64, int iWord);

    /// set the EventNr value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setEventNrWord64(boost::uint64_t& word64, int iWord);


    /// get/set technical trigger bits
    inline const TechnicalTriggerWord gtTechnicalTriggerWord() const
    {
        return m_gtTechnicalTriggerWord;
    }

    void setGtTechnicalTriggerWord (TechnicalTriggerWord gtTechnicalTriggerWordValue)
    {
        m_gtTechnicalTriggerWord = gtTechnicalTriggerWordValue;
    }

    void printGtTechnicalTriggerWord(std::ostream& myCout) const;

    /// set the technical trigger bits from a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setGtTechnicalTriggerWord(const boost::uint64_t& word64, int iWord);

    /// set the technical trigger bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setGtTechnicalTriggerWordWord64(boost::uint64_t& word64, int iWord);



    /// get/set/print algorithms bits (decision word)
    inline const DecisionWord gtDecisionWord() const
    {
        return m_gtDecisionWord;
    }

    void setGtDecisionWord(DecisionWord gtDecisionWordValue)
    {
        m_gtDecisionWord = gtDecisionWordValue;
    }

    void printGtDecisionWord(std::ostream& myCout) const;

    /// set the algorithms bits from two 64-bits words, having the first index iWord
    /// in the GTFE raw record
    /// WordA: bits 0-63
    void setGtDecisionWordA(const boost::uint64_t& word64, int iWord);

    /// set the algorithms bits from two 64-bits words, having the first index iWord
    /// in the GTFE raw record
    /// WordB: bits 64-128
    void setGtDecisionWordB(const boost::uint64_t& word64, int iWord);

    /// set the algorithms bits in two 64-bits word, having the first index iWord
    /// in the GTFE raw record
    /// WordA: bits 0-63
    void setGtDecisionWordAWord64(boost::uint64_t& word64, int iWord);

    /// set the algorithms bits in two 64-bits word, having the first index iWord
    /// in the GTFE raw record
    /// WordB: bits 64-128
    void setGtDecisionWordBWord64(boost::uint64_t& word64, int iWord);


    /// get/set extended algorithms bits (extended decision word)
    inline const DecisionWordExtended gtDecisionWordExtended() const
    {
        return m_gtDecisionWordExtended;
    }

    void setGtDecisionWordExtended (DecisionWordExtended gtDecisionWordExtendedValue)
    {
        m_gtDecisionWordExtended = gtDecisionWordExtendedValue;
    }

    void printGtDecisionWordExtended(std::ostream& myCout) const;

    /// set the extended algorithms bits from a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setGtDecisionWordExtended(const boost::uint64_t& word64, int iWord);

    /// set the extended algorithms bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setGtDecisionWordExtendedWord64(boost::uint64_t& word64, int iWord);



    /// get/set NoAlgo
    inline const boost::uint16_t noAlgo() const
    {
        return m_noAlgo;
    }

    void setNoAlgo(boost::uint16_t noAlgoValue)
    {
        m_noAlgo = noAlgoValue;
    }

    /// set the NoAlgo from a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setNoAlgo(const boost::uint64_t& word64, int iWord);

    /// set the NoAlgo bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setNoAlgoWord64(boost::uint64_t& word64, int iWord);



    /// get/set "Final OR" bits
    inline const boost::uint16_t finalOR() const
    {
        return m_finalOR;
    }

    void setFinalOR(boost::uint16_t finalORValue)
    {
        m_finalOR = finalORValue;
    }

    /// set the "Final OR" bits from a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setFinalOR(const boost::uint64_t& word64, int iWord);

    /// set the "Final OR" bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setFinalORWord64(boost::uint64_t& word64, int iWord);



    inline const bool globalDecision() const
    {
        return static_cast<bool> (m_finalOR);
    }

    /// get/set local bunch cross number of the actual bx
    inline const boost::uint16_t localBxNr() const
    {
        return m_localBxNr;
    }

    void setLocalBxNr(boost::uint16_t localBxNrValue)
    {
        m_localBxNr = localBxNrValue;
    }

    /// set the local bunch cross number bits from a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setLocalBxNr(const boost::uint64_t& word64, int iWord);

    /// set the local bunch cross number bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setLocalBxNrWord64(boost::uint64_t& word64, int iWord);



    /// get the size of the FDL block in GT DAQ record (in multiple of 8 bits)
    inline const unsigned int getSize() const
    {
        int unitLengthBits = L1GlobalTriggerReadoutSetup::UnitLength;

        return BlockSize*unitLengthBits;
    }

public:

    /// reset the content of a L1GtFdlWord
    void reset();

    /// pretty print the content of a L1GtFdlWord
    void print(std::ostream& myCout) const;


private:

    // block description in the raw GT record

    // block size in 64bits words (BlockSize * 64 bits)
    static const int BlockSize = 7;

    // word 0

    // index of the word in the FDL block containig the variable
    static const int BoardIdWord = 0;
    static const int BxInEventWord = 0;
    static const int BxNrWord = 0;
    static const int EventNrWord = 0;

    // mask to get the 64-bit-value from the corresponding word in the FDL block
    static const boost::uint64_t BoardIdMask =   0xFFFF000000000000ULL;
    static const boost::uint64_t BxInEventMask = 0x0000F00000000000ULL;
    static const boost::uint64_t BxNrMask =      0x00000FFF00000000ULL;
    static const boost::uint64_t EventNrMask =   0x0000000000FFFFFFULL;

    // shift to the right to get the value from the "64-bit-value"
    static const int BoardIdShift = 48;
    static const int BxInEventShift = 44;
    static const int BxNrShift = 32;
    static const int EventNrShift = 0;

    // word 1

    static const int GtTechnicalTriggerWordWord = 1;
    static const boost::uint64_t GtTechnicalTriggerWordMask = 0xFFFFFFFFFFFFFFFFULL;
    static const int GtTechnicalTriggerWordShift = 0;

    // word 2 - WordA: bits 0-63

    static const int GtDecisionWordAWord = 2;
    static const boost::uint64_t GtDecisionWordAMask = 0xFFFFFFFFFFFFFFFFULL;
    static const int GtDecisionWordAShift = 0;

    // word 3 - WordB: bits 64-128

    static const int GtDecisionWordBWord = 3;
    static const boost::uint64_t GtDecisionWordBMask = 0xFFFFFFFFFFFFFFFFULL;
    static const int GtDecisionWordBShift = 0;


    // word 4
    static const int GtDecisionWordExtendedWord = 4;
    static const boost::uint64_t GtDecisionWordExtendedMask = 0xFFFFFFFFFFFFFFFFULL;
    static const int GtDecisionWordExtendedShift = 0;

    // word 5
    static const int NoAlgoWord = 5;
    static const int FinalORWord = 5;

    static const boost::uint64_t NoAlgoMask =  0x0000000000000100ULL;
    static const boost::uint64_t FinalORMask = 0x00000000000000FFULL;

    static const int NoAlgoShift = 8;
    static const int FinalORShift = 0;

    // word 6
    static const int LocalBxNrWord = 6;
    static const boost::uint64_t LocalBxNrMask =  0x0000000000000FFFULL;
    static const int LocalBxNrShift = 0;

private:


    /// board identifier
    boost::uint16_t m_boardId;

    /// bunch cross in the GT event record
    int m_bxInEvent;

    /// bunch cross number of the actual bx
    boost::uint16_t m_bxNr;

    /// event number since last L1 reset generated in FDL
    boost::uint32_t m_eventNr;

    //

    /// technical trigger bits
    TechnicalTriggerWord m_gtTechnicalTriggerWord;

    /// algorithm bits
    DecisionWord m_gtDecisionWord;

    /// extended algorithm bits, in addition to 128
    DecisionWordExtended m_gtDecisionWordExtended;

    //

    // TODO meaning; just one bit?
    boost::uint16_t m_noAlgo;

    /// FINOR (7:0) Final OR bits.
    /// FINOR(i) is connected to Partition (i);
    boost::uint16_t m_finalOR;

    //

    /// local bunch cross number of the actual bx
    /// bx number at which the data were written into the ringbuffer
    boost::uint16_t m_localBxNr;


};

#endif /*L1GlobalTrigger_L1GtFdlWord_h*/
