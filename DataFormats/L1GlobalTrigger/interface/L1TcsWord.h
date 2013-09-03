#ifndef L1GlobalTrigger_L1TcsWord_h
#define L1GlobalTrigger_L1TcsWord_h

/**
 * \class L1TcsWord
 * 
 * 
 * Description: L1 Global Trigger - TCS words in the readout record.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// system include files
#include <iosfwd>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "FWCore/Utilities/interface/typedefs.h"

// forward declarations

// class declaration
class L1TcsWord
{

public:

    /// constructors
    L1TcsWord();    // empty constructor, all members set to zero;

    /// constructor from unpacked values;
    L1TcsWord(
        cms_uint16_t boardIdValue,
        cms_uint16_t bxNrValue,
        cms_uint16_t daqNrValue,
        cms_uint16_t triggerTypeValue,
        cms_uint16_t statusValue,
        cms_uint16_t luminositySegmentNrValue,
        cms_uint32_t partRunNrValue,
        cms_uint32_t assignedPartitionsValue,
        cms_uint32_t partTrigNrValue,
        cms_uint32_t eventNrValue,
        cms_uint64_t orbitNrValue );


    /// destructor
    virtual ~L1TcsWord();

    /// equal operator
    bool operator==(const L1TcsWord&) const;

    /// unequal operator
    bool operator!=(const L1TcsWord&) const;

public:

    /// get/set board ID

    /// get BoardId value
    inline const cms_uint16_t boardId() const
    {
        return m_boardId;
    }

    /// set BoardId from a BoardId value
    void setBoardId(const cms_uint16_t boardIdValue)
    {
        m_boardId = boardIdValue;
    }

    /// set the BoardId value from a 64-bits word
    void setBoardId(const cms_uint64_t& word64);

    /// set the BoardId value in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setBoardIdWord64(cms_uint64_t& word64, int iWord);



    /// get/set bunch cross number as counted in the TCS chip
    inline const cms_uint16_t bxNr() const
    {
        return m_bxNr;
    }

    void setBxNr(const cms_uint16_t bxNrValue)
    {
        m_bxNr = bxNrValue;
    }

    /// set the BxNr value from a 64-bits word
    void setBxNr(const cms_uint64_t& word64);

    /// set the BxNr value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setBxNrWord64(cms_uint64_t& word64, int iWord);



    /// get/set number of DAQ partition to which the L1A has been sent
    inline const cms_uint16_t daqNr() const
    {
        return m_daqNr;
    }

    void setDaqNr(const cms_uint16_t daqNrValue)
    {
        m_daqNr = daqNrValue;
    }

    /// set the DaqNr value from a 64-bits word
    void setDaqNr(const cms_uint64_t& word64);

    /// set the DaqNr value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setDaqNrWord64(cms_uint64_t& word64, int iWord);



    /// get/set trigger type, identical with event type in CMS header
    inline const cms_uint16_t triggerType() const
    {
        return m_triggerType;
    }

    void setTriggerType(const cms_uint16_t triggerTypeValue)
    {
        m_triggerType = triggerTypeValue;
    }

    /// set the TriggerType value from a 64-bits word
    void setTriggerType(const cms_uint64_t& word64);

    /// set the TriggerType value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setTriggerTypeWord64(cms_uint64_t& word64, int iWord);



    /// get/set status: 0000 = normal rate; 1000 = low rate = warning
    inline const cms_uint16_t status() const
    {
        return m_status;
    }

    void setStatus(const cms_uint16_t statusValue)
    {
        m_status = statusValue;
    }

    /// set the Status value from a 64-bits word
    void setStatus(const cms_uint64_t& word64);

    /// set the Status value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setStatusWord64(cms_uint64_t& word64, int iWord);


    /// get/set luminosity segment number
    inline const cms_uint16_t luminositySegmentNr() const
    {
        return m_luminositySegmentNr;
    }

    void setLuminositySegmentNr(const cms_uint16_t luminositySegmentNrValue)
    {
        m_luminositySegmentNr = luminositySegmentNrValue;
    }

    /// set the luminosity segment number value from a 64-bits word
    void setLuminositySegmentNr(const cms_uint64_t& word64);

    /// set the luminosity segment number value in a 64-bits word,
    /// having the index iWord in the GTFE EVM raw record
    void setLuminositySegmentNrWord64(cms_uint64_t& word64, int iWord);



    /// get/set partition run number
    inline const cms_uint32_t partRunNr() const
    {
        return m_partRunNr;
    }

    void setPartRunNr(const cms_uint32_t partRunNrValue)
    {
        m_partRunNr = partRunNrValue;
    }

    /// set the PartRunNr value from a 64-bits word
    void setPartRunNr(const cms_uint64_t& word64);

    /// set the PartRunNr value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setPartRunNrWord64(cms_uint64_t& word64, int iWord);



    /// get/set assigned partition: bit "i" correspond to detector partition "i"
    inline const cms_uint32_t assignedPartitions() const
    {
        return m_assignedPartitions;
    }

    void setAssignedPartitions(const cms_uint32_t assignedPartitionsValue)
    {
        m_assignedPartitions = assignedPartitionsValue;
    }

    /// set the AssignedPartitions value from a 64-bits word
    void setAssignedPartitions(const cms_uint64_t& word64);

    /// set the AssignedPartitions value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setAssignedPartitionsWord64(cms_uint64_t& word64, int iWord);



    /// get/set total number of L1A sent since start of the run to this DAQ partition
    inline const cms_uint32_t partTrigNr() const
    {
        return m_partTrigNr;
    }

    void setPartTrigNr(const cms_uint32_t partTrigNrValue)
    {
        m_partTrigNr = partTrigNrValue;
    }

    /// set the PartTrigNr value from a 64-bits word
    void setPartTrigNr(const cms_uint64_t& word64);

    /// set the PartTrigNr value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setPartTrigNrWord64(cms_uint64_t& word64, int iWord);



    /// get/set event number since last L1 reset generated in TCS chip
    inline const cms_uint32_t eventNr() const
    {
        return m_eventNr;
    }

    void setEventNr(const cms_uint32_t eventNrValue)
    {
        m_eventNr = eventNrValue;
    }

    /// set the EventNr value from a 64-bits word
    void setEventNr(const cms_uint64_t& word64);

    /// set the EventNr value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setEventNrWord64(cms_uint64_t& word64, int iWord);



    /// get/set orbit number since start of run
    inline const cms_uint64_t orbitNr() const
    {
        return m_orbitNr;
    }

    void setOrbitNr(const cms_uint64_t orbitNrValue)
    {
        m_orbitNr = orbitNrValue;
    }

    /// set the OrbitNr value from a 64-bits word
    void setOrbitNrFrom(const cms_uint64_t& word64);

    /// set the OrbitNr value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setOrbitNrWord64(cms_uint64_t& word64, int iWord);


    /// get the size of the TCS block in GT EVM record (in multiple of 8 bits)
    inline const unsigned int getSize() const
    {
        int unitLengthBits = L1GlobalTriggerReadoutSetup::UnitLength;

        return BlockSize*unitLengthBits;
    }

public:

    /// reset the content of a L1TcsWord
    void reset();

    /// pretty print the content of a L1TcsWord
    void print(std::ostream& myCout) const;

    /// unpack TCS
    /// tcsPtr pointer to the beginning of the TCS block in the raw data
    void unpack(const unsigned char* tcsPtr);



private:

    // block description in the raw GT record

    // block size in 64bits words (BlockSize * 64 bits)
    static const int BlockSize;

    // word 0

    // index of the word in the TCS block containig the variable
    static const int BoardIdWord;
    static const int BxNrWord;
    static const int DaqNrWord;
    static const int TriggerTypeWord;
    static const int StatusWord;
    static const int LuminositySegmentNrWord;

    // mask to get the 64-bit-value from the corresponding word in the TCS block
    static const cms_uint64_t BoardIdMask;
    static const cms_uint64_t BxNrMask;
    static const cms_uint64_t DaqNrMask;
    static const cms_uint64_t TriggerTypeMask;
    static const cms_uint64_t StatusMask;
    static const cms_uint64_t LuminositySegmentNrMask;

    // shift to the right to get the value from the "64-bit-value"
    static const int BoardIdShift;
    static const int BxNrShift;
    static const int DaqNrShift;
    static const int TriggerTypeShift;
    static const int StatusShift;
    static const int LuminositySegmentNrShift;

    // word 1

    static const int PartRunNrWord;
    static const int AssignedPartitionsWord;

    static const cms_uint64_t PartRunNrMask;
    static const cms_uint64_t AssignedPartitionsMask;

    static const int PartRunNrShift;
    static const int AssignedPartitionsShift;

    // word 2

    static const int PartTrigNrWord;
    static const int EventNrWord;

    static const cms_uint64_t PartTrigNrMask;
    static const cms_uint64_t EventNrMask;

    static const int PartTrigNrShift;
    static const int EventNrShift;

    // word 3

    static const int OrbitNrWord;

    static const cms_uint64_t OrbitNrMask;

    static const int OrbitNrShift;


    // word 4

    // empty



private:

    /// board identifier
    cms_uint16_t m_boardId;

    /// bunch cross number as counted in the TCS chip
    cms_uint16_t m_bxNr;

    /// number of DAQ partition to which the L1A has been sent
    cms_uint16_t m_daqNr;

    /// trigger type, identical with event type in CMS header (event type)
    cms_uint16_t m_triggerType;

    /// 0000 = normal rate; 1000 = low rate = warning
    cms_uint16_t m_status;

    /// luminosity segment number
    cms_uint16_t m_luminositySegmentNr;

    //

    /// partition run number
    cms_uint32_t m_partRunNr;

    /// bit "i" correspond to detector partition "i"
    /// if bit = 1, detection partition connected to actual
    /// DAQ partition
    cms_uint32_t m_assignedPartitions;

    //

    /// total number of L1A sent since start of the run
    /// to this DAQ partition
    cms_uint32_t m_partTrigNr;

    /// event number since last L1 reset generated in TCS chip
    cms_uint32_t m_eventNr;

    //

    /// orbit number since start of run (48 bits, in fact)
    cms_uint64_t m_orbitNr;

    //
    // empty word

};

#endif /*L1GlobalTrigger_L1TcsWord_h*/
