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
        boost::uint16_t gtPrescaleFactorIndexTechValue,
        boost::uint16_t gtPrescaleFactorIndexAlgoValue,
        boost::uint16_t noAlgoValue,
        boost::uint16_t finalORValue,
        boost::uint32_t orbitNrValue,
        boost::uint16_t lumiSegmentNrValue,
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
    void setBoardId(const boost::uint16_t& boardIdValue)
    {
        m_boardId = boardIdValue;
    }

    /// set the BoardId value from a 64-bits word
    void setBoardId(const boost::uint64_t& word64);

    /// set the BoardId value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setBoardIdWord64(boost::uint64_t& word64, const int iWord);



    /// get/set bunch cross in the GT event record
    inline const int bxInEvent() const
    {
        return m_bxInEvent;
    }

    void setBxInEvent(const int bxInEventValue)
    {
        m_bxInEvent = bxInEventValue;
    }

    /// set the BxInEvent value from a 64-bits word
    void setBxInEvent(const boost::uint64_t& word64);

    /// set the BxInEvent value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setBxInEventWord64(boost::uint64_t& word64, const int iWord);



    /// get/set BxNr - bunch cross number of the actual bx
    inline const boost::uint16_t bxNr() const
    {
        return m_bxNr;
    }

    void setBxNr(const boost::uint16_t& bxNrValue)
    {
        m_bxNr = bxNrValue;
    }

    /// set the BxNr value from a 64-bits word
    void setBxNr(const boost::uint64_t& word64);

    /// set the BxNr value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setBxNrWord64(boost::uint64_t& word64, const int iWord);



    /// get/set event number since last L1 reset generated in FDL
    inline const boost::uint32_t eventNr() const
    {
        return m_eventNr;
    }

    void setEventNr(const boost::uint32_t& eventNrValue)
    {
        m_eventNr = eventNrValue;
    }

    /// set the EventNr value from a 64-bits word
    void setEventNr(const boost::uint64_t& word64);

    /// set the EventNr value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setEventNrWord64(boost::uint64_t& word64, const int iWord);


    /// get/set technical trigger bits
    inline const TechnicalTriggerWord & gtTechnicalTriggerWord() const
    {
        return m_gtTechnicalTriggerWord;
    }

    void setGtTechnicalTriggerWord (const TechnicalTriggerWord& gtTechnicalTriggerWordValue)
    {
        m_gtTechnicalTriggerWord = gtTechnicalTriggerWordValue;
    }

    void printGtTechnicalTriggerWord(std::ostream& myCout) const;

    /// set the technical trigger bits from a 64-bits word
    void setGtTechnicalTriggerWord(const boost::uint64_t& word64);

    /// set the technical trigger bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setGtTechnicalTriggerWordWord64(boost::uint64_t& word64, const int iWord);



    /// get/set/print algorithms bits (decision word)
    inline const DecisionWord & gtDecisionWord() const
    {
        return m_gtDecisionWord;
    }

    void setGtDecisionWord(const DecisionWord& gtDecisionWordValue)
    {
        m_gtDecisionWord = gtDecisionWordValue;
    }

    void printGtDecisionWord(std::ostream& myCout) const;

    /// set the algorithms bits from two 64-bits words
    /// WordA: bits 0-63
    void setGtDecisionWordA(const boost::uint64_t& word64);

    /// set the algorithms bits from two 64-bits words
    /// WordB: bits 64-128
    void setGtDecisionWordB(const boost::uint64_t& word64);

    /// set the algorithms bits in two 64-bits word, having the first index iWord
    /// in the GTFE raw record
    /// WordA: bits 0-63
    void setGtDecisionWordAWord64(boost::uint64_t& word64, const int iWord);

    /// set the algorithms bits in two 64-bits word, having the first index iWord
    /// in the GTFE raw record
    /// WordB: bits 64-128
    void setGtDecisionWordBWord64(boost::uint64_t& word64, const int iWord);


    /// get/set extended algorithms bits (extended decision word)
    inline const DecisionWordExtended & gtDecisionWordExtended() const
    {
        return m_gtDecisionWordExtended;
    }

    void setGtDecisionWordExtended (const DecisionWordExtended& gtDecisionWordExtendedValue)
    {
        m_gtDecisionWordExtended = gtDecisionWordExtendedValue;
    }

    void printGtDecisionWordExtended(std::ostream& myCout) const;

    /// set the extended algorithms bits from a 64-bits word
    void setGtDecisionWordExtended(const boost::uint64_t& word64);

    /// set the extended algorithms bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setGtDecisionWordExtendedWord64(boost::uint64_t& word64, const int iWord);





    /// get/set "physics declared" bit
    inline const boost::uint16_t physicsDeclared() const
    {
        return m_physicsDeclared;
    }

    void setPhysicsDeclared(const boost::uint16_t& physicsDeclaredValue)
    {
        m_physicsDeclared = physicsDeclaredValue;
    }

    /// set the "physics declared" bit from a 64-bits word
    void setPhysicsDeclared(const boost::uint64_t& word64);

    /// set the "physics declared" bit bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setPhysicsDeclaredWord64(boost::uint64_t& word64, const int iWord);



    /// get/set index of the set of prescale factors

    inline const boost::uint16_t gtPrescaleFactorIndexTech() const
    {
        return m_gtPrescaleFactorIndexTech;
    }

    void setGtPrescaleFactorIndexTech(const boost::uint16_t& gtPrescaleFactorIndexTechValue)
    {
        m_gtPrescaleFactorIndexTech = gtPrescaleFactorIndexTechValue;
    }

    /// set the GtPrescaleFactorIndexTech from a 64-bits word
    void setGtPrescaleFactorIndexTech(const boost::uint64_t& word64);

    /// set the GtPrescaleFactorIndexTech bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setGtPrescaleFactorIndexTechWord64(boost::uint64_t& word64, const int iWord);



    inline const boost::uint16_t gtPrescaleFactorIndexAlgo() const
    {
        return m_gtPrescaleFactorIndexAlgo;
    }

    void setGtPrescaleFactorIndexAlgo(const boost::uint16_t& gtPrescaleFactorIndexAlgoValue)
    {
        m_gtPrescaleFactorIndexAlgo = gtPrescaleFactorIndexAlgoValue;
    }

    /// set the GtPrescaleFactorIndexAlgo from a 64-bits word
    void setGtPrescaleFactorIndexAlgo(const boost::uint64_t& word64);

    /// set the GtPrescaleFactorIndexAlgo bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setGtPrescaleFactorIndexAlgoWord64(boost::uint64_t& word64, const int iWord);


    /// get/set NoAlgo
    inline const boost::uint16_t noAlgo() const
    {
        return m_noAlgo;
    }

    void setNoAlgo(const boost::uint16_t& noAlgoValue)
    {
        m_noAlgo = noAlgoValue;
    }

    /// set the NoAlgo from a 64-bits word
    void setNoAlgo(const boost::uint64_t& word64);

    /// set the NoAlgo bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setNoAlgoWord64(boost::uint64_t& word64, const int iWord);



    /// get/set "Final OR" bits
    inline const boost::uint16_t finalOR() const
    {
        return m_finalOR;
    }

    void setFinalOR(const boost::uint16_t& finalORValue)
    {
        m_finalOR = finalORValue;
    }

    /// set the "Final OR" bits from a 64-bits word
    void setFinalOR(const boost::uint64_t& word64);

    /// set the "Final OR" bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setFinalORWord64(boost::uint64_t& word64, const int iWord);



    inline const bool globalDecision() const
    {
        return static_cast<bool> (m_finalOR);
    }


    /// get/set orbit number
    inline const boost::uint32_t orbitNr() const
    {
        return m_orbitNr;
    }

    void setOrbitNr(const boost::uint32_t& orbitNrValue)
    {
        m_orbitNr = orbitNrValue;
    }

    /// set the orbit number bits from a 64-bits word
    void setOrbitNr(const boost::uint64_t& word64);

    /// set the orbit number bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setOrbitNrWord64(boost::uint64_t& word64, const int iWord);



    /// get/set luminosity segment number of the actual bx
    inline const boost::uint16_t lumiSegmentNr() const
    {
        return m_lumiSegmentNr;
    }

    void setLumiSegmentNr(const boost::uint16_t& lumiSegmentNrValue)
    {
        m_lumiSegmentNr = lumiSegmentNrValue;
    }

    /// set the luminosity segment number bits from a 64-bits word
    void setLumiSegmentNr(const boost::uint64_t& word64);

    /// set the luminosity segment number bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setLumiSegmentNrWord64(boost::uint64_t& word64, const int iWord);



    /// get/set local bunch cross number of the actual bx
    inline const boost::uint16_t localBxNr() const
    {
        return m_localBxNr;
    }

    void setLocalBxNr(const boost::uint16_t& localBxNrValue)
    {
        m_localBxNr = localBxNrValue;
    }

    /// set the local bunch cross number bits from a 64-bits word
    void setLocalBxNr(const boost::uint64_t& word64);

    /// set the local bunch cross number bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setLocalBxNrWord64(boost::uint64_t& word64, const int iWord);



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

    /// unpack FDL
    /// fdlPtr pointer to the beginning of the FDL block in the raw data
    void unpack(const unsigned char* fdlPtr);

private:

    // block description in the raw GT record

    // block size in 64bits words (BlockSize * 64 bits)
    static const int BlockSize;

    // word 0

    // index of the word in the FDL block containig the variable
    static const int BoardIdWord;
    static const int BxInEventWord;
    static const int BxNrWord;
    static const int EventNrWord;

    // mask to get the 64-bit-value from the corresponding word in the FDL block
    static const boost::uint64_t BoardIdMask;
    static const boost::uint64_t BxInEventMask;
    static const boost::uint64_t BxNrMask;
    static const boost::uint64_t EventNrMask;

    // shift to the right to get the value from the "64-bit-value"
    static const int BoardIdShift;
    static const int BxInEventShift;
    static const int BxNrShift;
    static const int EventNrShift;

    // word 1

    static const int GtTechnicalTriggerWordWord;
    static const boost::uint64_t GtTechnicalTriggerWordMask;
    static const int GtTechnicalTriggerWordShift;

    // word 2 - WordA: bits 0-63

    static const int GtDecisionWordAWord;
    static const boost::uint64_t GtDecisionWordAMask;
    static const int GtDecisionWordAShift;

    // word 3 - WordB: bits 64-128

    static const int GtDecisionWordBWord;
    static const boost::uint64_t GtDecisionWordBMask;
    static const int GtDecisionWordBShift;


    // word 4
    static const int GtDecisionWordExtendedWord;
    static const boost::uint64_t GtDecisionWordExtendedMask;
    static const int GtDecisionWordExtendedShift;

    // word 5
    static const int PhysicsDeclaredWord;
    static const int GtPrescaleFactorIndexTechWord;
    static const int GtPrescaleFactorIndexAlgoWord;
    static const int NoAlgoWord;
    static const int FinalORWord;

    static const boost::uint64_t PhysicsDeclaredMask;
    static const boost::uint64_t GtPrescaleFactorIndexTechMask;
    static const boost::uint64_t GtPrescaleFactorIndexAlgoMask;
    static const boost::uint64_t NoAlgoMask;
    static const boost::uint64_t FinalORMask;

    static const int PhysicsDeclaredShift;
    static const int GtPrescaleFactorIndexTechShift;
    static const int GtPrescaleFactorIndexAlgoShift;
    static const int NoAlgoShift;
    static const int FinalORShift;

    // word 6
    static const int OrbitNrWord;
    static const int LumiSegmentNrWord;
    static const int LocalBxNrWord;

    static const boost::uint64_t OrbitNrMask;
    static const boost::uint64_t LumiSegmentNrMask;
    static const boost::uint64_t LocalBxNrMask;

    static const int OrbitNrShift;
    static const int LumiSegmentNrShift;
    static const int LocalBxNrShift;

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
    /// set to true when physics declared
    boost::uint16_t m_physicsDeclared;

    /// index of the set of prescale factors in the DB/EventSetup
    /// for algorithm triggers and technical triggers
    boost::uint16_t m_gtPrescaleFactorIndexTech;
    boost::uint16_t m_gtPrescaleFactorIndexAlgo;

    /// true if no algorithm (from a defined group - default all) triggered
    boost::uint16_t m_noAlgo;

    /// FINOR (7:0) Final OR bits.
    /// FINOR(i) is connected to Partition (i);
    boost::uint16_t m_finalOR;

    //

    /// orbit number
    boost::uint32_t m_orbitNr;

    /// luminosity segment number
    boost::uint16_t m_lumiSegmentNr;

    /// local bunch cross number of the actual bx
    /// bx number at which the data were written into the ringbuffer
    boost::uint16_t m_localBxNr;



};

#endif /*L1GlobalTrigger_L1GtFdlWord_h*/
