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

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "FWCore/Utilities/interface/typedefs.h"

// forward declarations

// class interface

class L1GtFdlWord
{

public:
    /// constructors
    L1GtFdlWord();    // empty constructor, all members set to zero;

    // constructor from unpacked values;
    L1GtFdlWord(
        cms_uint16_t boardIdValue,
        int bxInEventValue,
        cms_uint16_t bxNrValue,
        cms_uint32_t eventNrValue,
        const TechnicalTriggerWord& gtTechnicalTriggerWordValue,
        const DecisionWord& gtDecisionWordValue,
        const DecisionWordExtended& gtDecisionWordExtendedValue,
        cms_uint16_t gtPrescaleFactorIndexTechValue,
        cms_uint16_t gtPrescaleFactorIndexAlgoValue,
        cms_uint16_t noAlgoValue,
        cms_uint16_t finalORValue,
        cms_uint32_t orbitNrValue,
        cms_uint16_t lumiSegmentNrValue,
        cms_uint16_t localBxNrValue
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
    inline const cms_uint16_t boardId() const
    {
        return m_boardId;
    }

    /// set BoardId from a BoardId value
    void setBoardId(const cms_uint16_t& boardIdValue)
    {
        m_boardId = boardIdValue;
    }

    /// set the BoardId value from a 64-bits word
    void setBoardId(const cms_uint64_t& word64);

    /// set the BoardId value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setBoardIdWord64(cms_uint64_t& word64, const int iWord);



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
    void setBxInEvent(const cms_uint64_t& word64);

    /// set the BxInEvent value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setBxInEventWord64(cms_uint64_t& word64, const int iWord);



    /// get/set BxNr - bunch cross number of the actual bx
    inline const cms_uint16_t bxNr() const
    {
        return m_bxNr;
    }

    void setBxNr(const cms_uint16_t& bxNrValue)
    {
        m_bxNr = bxNrValue;
    }

    /// set the BxNr value from a 64-bits word
    void setBxNr(const cms_uint64_t& word64);

    /// set the BxNr value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setBxNrWord64(cms_uint64_t& word64, const int iWord);



    /// get/set event number since last L1 reset generated in FDL
    inline const cms_uint32_t eventNr() const
    {
        return m_eventNr;
    }

    void setEventNr(const cms_uint32_t& eventNrValue)
    {
        m_eventNr = eventNrValue;
    }

    /// set the EventNr value from a 64-bits word
    void setEventNr(const cms_uint64_t& word64);

    /// set the EventNr value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setEventNrWord64(cms_uint64_t& word64, const int iWord);


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
    void setGtTechnicalTriggerWord(const cms_uint64_t& word64);

    /// set the technical trigger bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setGtTechnicalTriggerWordWord64(cms_uint64_t& word64, const int iWord);



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
    void setGtDecisionWordA(const cms_uint64_t& word64);

    /// set the algorithms bits from two 64-bits words
    /// WordB: bits 64-128
    void setGtDecisionWordB(const cms_uint64_t& word64);

    /// set the algorithms bits in two 64-bits word, having the first index iWord
    /// in the GTFE raw record
    /// WordA: bits 0-63
    void setGtDecisionWordAWord64(cms_uint64_t& word64, const int iWord);

    /// set the algorithms bits in two 64-bits word, having the first index iWord
    /// in the GTFE raw record
    /// WordB: bits 64-128
    void setGtDecisionWordBWord64(cms_uint64_t& word64, const int iWord);


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
    void setGtDecisionWordExtended(const cms_uint64_t& word64);

    /// set the extended algorithms bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setGtDecisionWordExtendedWord64(cms_uint64_t& word64, const int iWord);





    /// get/set "physics declared" bit
    inline const cms_uint16_t physicsDeclared() const
    {
        return m_physicsDeclared;
    }

    void setPhysicsDeclared(const cms_uint16_t& physicsDeclaredValue)
    {
        m_physicsDeclared = physicsDeclaredValue;
    }

    /// set the "physics declared" bit from a 64-bits word
    void setPhysicsDeclared(const cms_uint64_t& word64);

    /// set the "physics declared" bit bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setPhysicsDeclaredWord64(cms_uint64_t& word64, const int iWord);



    /// get/set index of the set of prescale factors

    inline const cms_uint16_t gtPrescaleFactorIndexTech() const
    {
        return m_gtPrescaleFactorIndexTech;
    }

    void setGtPrescaleFactorIndexTech(const cms_uint16_t& gtPrescaleFactorIndexTechValue)
    {
        m_gtPrescaleFactorIndexTech = gtPrescaleFactorIndexTechValue;
    }

    /// set the GtPrescaleFactorIndexTech from a 64-bits word
    void setGtPrescaleFactorIndexTech(const cms_uint64_t& word64);

    /// set the GtPrescaleFactorIndexTech bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setGtPrescaleFactorIndexTechWord64(cms_uint64_t& word64, const int iWord);



    inline const cms_uint16_t gtPrescaleFactorIndexAlgo() const
    {
        return m_gtPrescaleFactorIndexAlgo;
    }

    void setGtPrescaleFactorIndexAlgo(const cms_uint16_t& gtPrescaleFactorIndexAlgoValue)
    {
        m_gtPrescaleFactorIndexAlgo = gtPrescaleFactorIndexAlgoValue;
    }

    /// set the GtPrescaleFactorIndexAlgo from a 64-bits word
    void setGtPrescaleFactorIndexAlgo(const cms_uint64_t& word64);

    /// set the GtPrescaleFactorIndexAlgo bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setGtPrescaleFactorIndexAlgoWord64(cms_uint64_t& word64, const int iWord);


    /// get/set NoAlgo
    inline const cms_uint16_t noAlgo() const
    {
        return m_noAlgo;
    }

    void setNoAlgo(const cms_uint16_t& noAlgoValue)
    {
        m_noAlgo = noAlgoValue;
    }

    /// set the NoAlgo from a 64-bits word
    void setNoAlgo(const cms_uint64_t& word64);

    /// set the NoAlgo bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setNoAlgoWord64(cms_uint64_t& word64, const int iWord);



    /// get/set "Final OR" bits
    inline const cms_uint16_t finalOR() const
    {
        return m_finalOR;
    }

    void setFinalOR(const cms_uint16_t& finalORValue)
    {
        m_finalOR = finalORValue;
    }

    /// set the "Final OR" bits from a 64-bits word
    void setFinalOR(const cms_uint64_t& word64);

    /// set the "Final OR" bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setFinalORWord64(cms_uint64_t& word64, const int iWord);



    inline const bool globalDecision() const
    {
        return static_cast<bool> (m_finalOR);
    }


    /// get/set orbit number
    inline const cms_uint32_t orbitNr() const
    {
        return m_orbitNr;
    }

    void setOrbitNr(const cms_uint32_t& orbitNrValue)
    {
        m_orbitNr = orbitNrValue;
    }

    /// set the orbit number bits from a 64-bits word
    void setOrbitNr(const cms_uint64_t& word64);

    /// set the orbit number bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setOrbitNrWord64(cms_uint64_t& word64, const int iWord);



    /// get/set luminosity segment number of the actual bx
    inline const cms_uint16_t lumiSegmentNr() const
    {
        return m_lumiSegmentNr;
    }

    void setLumiSegmentNr(const cms_uint16_t& lumiSegmentNrValue)
    {
        m_lumiSegmentNr = lumiSegmentNrValue;
    }

    /// set the luminosity segment number bits from a 64-bits word
    void setLumiSegmentNr(const cms_uint64_t& word64);

    /// set the luminosity segment number bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setLumiSegmentNrWord64(cms_uint64_t& word64, const int iWord);



    /// get/set local bunch cross number of the actual bx
    inline const cms_uint16_t localBxNr() const
    {
        return m_localBxNr;
    }

    void setLocalBxNr(const cms_uint16_t& localBxNrValue)
    {
        m_localBxNr = localBxNrValue;
    }

    /// set the local bunch cross number bits from a 64-bits word
    void setLocalBxNr(const cms_uint64_t& word64);

    /// set the local bunch cross number bits in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setLocalBxNrWord64(cms_uint64_t& word64, const int iWord);



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
    static const cms_uint64_t BoardIdMask;
    static const cms_uint64_t BxInEventMask;
    static const cms_uint64_t BxNrMask;
    static const cms_uint64_t EventNrMask;

    // shift to the right to get the value from the "64-bit-value"
    static const int BoardIdShift;
    static const int BxInEventShift;
    static const int BxNrShift;
    static const int EventNrShift;

    // word 1

    static const int GtTechnicalTriggerWordWord;
    static const cms_uint64_t GtTechnicalTriggerWordMask;
    static const int GtTechnicalTriggerWordShift;

    // word 2 - WordA: bits 0-63

    static const int GtDecisionWordAWord;
    static const cms_uint64_t GtDecisionWordAMask;
    static const int GtDecisionWordAShift;

    // word 3 - WordB: bits 64-128

    static const int GtDecisionWordBWord;
    static const cms_uint64_t GtDecisionWordBMask;
    static const int GtDecisionWordBShift;


    // word 4
    static const int GtDecisionWordExtendedWord;
    static const cms_uint64_t GtDecisionWordExtendedMask;
    static const int GtDecisionWordExtendedShift;

    // word 5
    static const int PhysicsDeclaredWord;
    static const int GtPrescaleFactorIndexTechWord;
    static const int GtPrescaleFactorIndexAlgoWord;
    static const int NoAlgoWord;
    static const int FinalORWord;

    static const cms_uint64_t PhysicsDeclaredMask;
    static const cms_uint64_t GtPrescaleFactorIndexTechMask;
    static const cms_uint64_t GtPrescaleFactorIndexAlgoMask;
    static const cms_uint64_t NoAlgoMask;
    static const cms_uint64_t FinalORMask;

    static const int PhysicsDeclaredShift;
    static const int GtPrescaleFactorIndexTechShift;
    static const int GtPrescaleFactorIndexAlgoShift;
    static const int NoAlgoShift;
    static const int FinalORShift;

    // word 6
    static const int OrbitNrWord;
    static const int LumiSegmentNrWord;
    static const int LocalBxNrWord;

    static const cms_uint64_t OrbitNrMask;
    static const cms_uint64_t LumiSegmentNrMask;
    static const cms_uint64_t LocalBxNrMask;

    static const int OrbitNrShift;
    static const int LumiSegmentNrShift;
    static const int LocalBxNrShift;

private:


    /// board identifier
    cms_uint16_t m_boardId;

    /// bunch cross in the GT event record
    int m_bxInEvent;

    /// bunch cross number of the actual bx
    cms_uint16_t m_bxNr;

    /// event number since last L1 reset generated in FDL
    cms_uint32_t m_eventNr;

    //

    /// technical trigger bits
    TechnicalTriggerWord m_gtTechnicalTriggerWord;

    /// algorithm bits
    DecisionWord m_gtDecisionWord;

    /// extended algorithm bits, in addition to 128
    DecisionWordExtended m_gtDecisionWordExtended;

    //
    /// set to true when physics declared
    cms_uint16_t m_physicsDeclared;

    /// index of the set of prescale factors in the DB/EventSetup
    /// for algorithm triggers and technical triggers
    cms_uint16_t m_gtPrescaleFactorIndexTech;
    cms_uint16_t m_gtPrescaleFactorIndexAlgo;

    /// true if no algorithm (from a defined group - default all) triggered
    cms_uint16_t m_noAlgo;

    /// FINOR (7:0) Final OR bits.
    /// FINOR(i) is connected to Partition (i);
    cms_uint16_t m_finalOR;

    //

    /// orbit number
    cms_uint32_t m_orbitNr;

    /// luminosity segment number
    cms_uint16_t m_lumiSegmentNr;

    /// local bunch cross number of the actual bx
    /// bx number at which the data were written into the ringbuffer
    cms_uint16_t m_localBxNr;



};

#endif /*L1GlobalTrigger_L1GtFdlWord_h*/
