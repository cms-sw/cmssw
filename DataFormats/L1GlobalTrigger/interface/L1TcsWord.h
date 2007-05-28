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
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <boost/cstdint.hpp>
#include <iostream>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

// forward declarations

// class declaration
class L1TcsWord
{

public:

    /// constructors
    L1TcsWord();    // empty constructor, all members set to zero;

    /// constructor from unpacked values;
    L1TcsWord(
        boost::uint16_t daqNrValue,
        boost::uint16_t triggerTypeValue,
        boost::uint16_t statusValue,
        boost::uint16_t bxNrValue,
        boost::uint32_t partTrigNrValue,
        boost::uint32_t eventNrValue,
        boost::uint32_t assignedPartitionsValue,
        boost::uint32_t partRunNrValue,
        boost::uint32_t orbitNrValue );


    /// destructor
    virtual ~L1TcsWord();

    /// equal operator
    bool operator==(const L1TcsWord&) const;

    /// unequal operator
    bool operator!=(const L1TcsWord&) const;

public:

    /// get/set number of DAQ partition to which the L1A has been sent
    inline const boost::uint16_t daqNr() const
    {
        return m_daqNr;
    }

    void setDaqNr(boost::uint16_t daqNrValue)
    {
        m_daqNr = daqNrValue;
    }

    /// set the DaqNr value from a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setDaqNr(const boost::uint64_t& word64, int iWord);

    /// set the DaqNr value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setDaqNrWord64(boost::uint64_t& word64, int iWord);

    /// get/set trigger type, identical with event type in CMS header
    inline const boost::uint16_t triggerType() const
    {
        return m_triggerType;
    }

    void setTriggerType(boost::uint16_t triggerTypeValue)
    {
        m_triggerType = triggerTypeValue;
    }

    /// set the TriggerType value from a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setTriggerType(const boost::uint64_t& word64, int iWord);

    /// set the TriggerType value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setTriggerTypeWord64(boost::uint64_t& word64, int iWord);


    /// get/set status: 0000 = normal rate; 1000 = low rate = warning
    inline const boost::uint16_t status() const
    {
        return m_status;
    }

    void setStatus(boost::uint16_t statusValue)
    {
        m_status = statusValue;
    }

    /// set the Status value from a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setStatus(const boost::uint64_t& word64, int iWord);

    /// set the Status value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setStatusWord64(boost::uint64_t& word64, int iWord);


    /// get/set bunch cross number as counted in the TCS chip
    inline const boost::uint16_t bxNr() const
    {
        return m_bxNr;
    }

    void setBxNr(boost::uint16_t bxNrValue)
    {
        m_bxNr = bxNrValue;
    }

    /// set the BxNr value from a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setBxNr(const boost::uint64_t& word64, int iWord);

    /// set the BxNr value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setBxNrWord64(boost::uint64_t& word64, int iWord);


    /// get/set total number of L1A sent since start of the run to this DAQ partition
    inline const boost::uint32_t partTrigNr() const
    {
        return m_partTrigNr;
    }

    void setPartTrigNr(boost::uint32_t partTrigNrValue)
    {
        m_partTrigNr = partTrigNrValue;
    }

    /// set the PartTrigNr value from a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setPartTrigNr(const boost::uint64_t& word64, int iWord);

    /// set the PartTrigNr value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setPartTrigNrWord64(boost::uint64_t& word64, int iWord);


    /// get/set event number since last L1 reset generated in TCS chip
    inline const boost::uint32_t eventNr() const
    {
        return m_eventNr;
    }

    void setEventNr(boost::uint32_t eventNrValue)
    {
        m_eventNr = eventNrValue;
    }

    /// set the EventNr value from a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setEventNr(const boost::uint64_t& word64, int iWord);

    /// set the EventNr value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setEventNrWord64(boost::uint64_t& word64, int iWord);

    /// get/set assigned partition: bit "i" correspond to detector partition "i"
    inline const boost::uint32_t assignedPartitions() const
    {
        return m_assignedPartitions;
    }

    void setAssignedPartitions(boost::uint32_t assignedPartitionsValue)
    {
        m_assignedPartitions = assignedPartitionsValue;
    }

    /// set the AssignedPartitions value from a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setAssignedPartitions(const boost::uint64_t& word64, int iWord);

    /// set the AssignedPartitions value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setAssignedPartitionsWord64(boost::uint64_t& word64, int iWord);


    /// get/set partition run number
    inline const boost::uint32_t partRunNr() const
    {
        return m_partRunNr;
    }

    void setPartRunNr(boost::uint32_t partRunNrValue)
    {
        m_partRunNr = partRunNrValue;
    }

    /// set the PartRunNr value from a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setPartRunNr(const boost::uint64_t& word64, int iWord);

    /// set the PartRunNr value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setPartRunNrWord64(boost::uint64_t& word64, int iWord);

    /// get/set orbit number since start of run
    inline const boost::uint32_t orbitNr() const
    {
        return m_orbitNr;
    }

    void setOrbitNr(boost::uint32_t orbitNrValue)
    {
        m_orbitNr = orbitNrValue;
    }

    /// set the OrbitNr value from a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setOrbitNr(const boost::uint64_t& word64, int iWord);

    /// set the OrbitNr value in a 64-bits word, having the index iWord
    /// in the GTFE EVM raw record
    void setOrbitNrWord64(boost::uint64_t& word64, int iWord);


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

private:

    // block description in the raw GT record

    // block size in 64bits words
    static const int BlockSize = 3;        // 3 x 64bits

    // word 0

    static const int DaqNrWord = 0;
    static const int TriggerTypeWord = 0;
    static const int StatusWord = 0;
    static const int BxNrWord = 0;
    static const int PartTrigNrWord = 0;

    static const boost::uint64_t DaqNrMask =       0x0F00000000000000ULL;
    static const boost::uint64_t TriggerTypeMask = 0x00F0000000000000ULL;
    static const boost::uint64_t StatusMask =      0x000F000000000000ULL;
    static const boost::uint64_t BxNrMask =        0x00000FFF00000000ULL;
    static const boost::uint64_t PartTrigNrMask =  0x00000000FFFFFFFFULL;

    // shifts could be computed from masks...
    static const int DaqNrShift = 56;
    static const int TriggerTypeShift = 52;
    static const int StatusShift = 48;
    static const int BxNrShift = 32;
    static const int PartTrigNrShift = 0;

    // word 1

    static const int EventNrWord = 1;
    static const int AssignedPartitionsWord = 1;

    static const boost::uint64_t EventNrMask =            0x00FFFFFF00000000ULL;
    static const boost::uint64_t AssignedPartitionsMask = 0x00000000FFFFFFFFULL;

    static const int EventNrShift = 32;
    static const int AssignedPartitionsShift = 0;

    // word 2

    static const int PartRunNrWord = 2;
    static const int OrbitNrWord =   2;

    static const boost::uint64_t PartRunNrMask = 0xFFFFFFFF00000000ULL;
    static const boost::uint64_t OrbitNrMask =   0x00000000FFFFFFFFULL;

    static const int PartRunNrShift = 32;
    static const int OrbitNrShift = 0;

private:

    /// first number in the comment represents number of bits

    ///  4: number of DAQ partition to which the L1A has been sent
    boost::uint16_t m_daqNr;

    ///  4: trigger type, identical with event type in CMS header
    boost::uint16_t m_triggerType;

    ///  4: 0000 = normal rate; 1000 = low rate = warning
    boost::uint16_t m_status;

    /// 12: bunch cross number as counted in the TCS chip
    boost::uint16_t m_bxNr;

    /// 32: total number of L1A sent since start of the run
    ///     to this DAQ partition
    ///     TODO overflow after 11.8h at 100 Hz
    boost::uint32_t m_partTrigNr;

    //

    /// 24: event number since last L1 reset generated in TCS chip
    boost::uint32_t m_eventNr;

    /// 32: bit "i" correspond to detector partition "i"
    ///     if bit = 1, detection partition connected to actual
    ///     DAQ partition
    boost::uint32_t m_assignedPartitions;

    //

    /// 32: TODO clarify meaning
    boost::uint32_t m_partRunNr;

    /// 32: orbit number since start of run
    boost::uint32_t m_orbitNr;

};

#endif /*L1GlobalTrigger_L1TcsWord_h*/
