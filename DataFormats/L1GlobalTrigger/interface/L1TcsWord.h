#ifndef L1GlobalTrigger_L1TcsWord_h
#define L1GlobalTrigger_L1TcsWord_h

/**
 * \class L1TcsWord
 * 
 * 
 * 
 * Description: L1 Global Trigger - TCS words in the readout record 
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
//   base class

// forward declarations

// class interface

class L1TcsWord
{

public:
    /// constructors
    L1TcsWord();    // empty constructor, all members set to zero;
    
    // constructor from unpacked values;
    L1TcsWord(
        uint16_t daqNrValue,
        uint16_t triggerTypeValue,
        uint16_t statusValue,
        uint16_t bxNrValue,
        uint32_t partTrigNrValue,
        uint32_t partRunNrValue,
        uint32_t eventNrValue, 
        uint32_t assignedPartitionsValue, 
        uint32_t orbitNrValue );
        

    /// destructor
    virtual ~L1TcsWord();

    /// equal operator
    bool operator==(const L1TcsWord&) const;

    /// unequal operator
    bool operator!=(const L1TcsWord&) const;

public:

    /// get/set number of DAQ partition to which the L1A has been sent
    inline const uint16_t daqNr() const { return m_daqNr; }
    void setDaqNr(uint16_t daqNrValue) { m_daqNr = daqNrValue; }

    /// get/set trigger type, identical with event type in CMS header
    inline const uint16_t triggerType() const { return m_triggerType; }
    void setTriggerType(uint16_t triggerTypeValue) { m_triggerType = triggerTypeValue; }

    /// get/set status: 0000 = normal rate; 1000 = low rate = warning
    inline const uint16_t status() const { return m_status; }
    void setStatus(uint16_t statusValue) { m_status = statusValue; }

    /// get/set bunch cross number as counted in the TCS chip
    inline const uint16_t bxNr() const { return m_bxNr; }
    void setBxNr(uint16_t bxNrValue) { m_bxNr = bxNrValue; }

    /// get/set total number of L1A sent since start of the run to this DAQ partition 
    inline const uint32_t partTrigNr() const { return m_partTrigNr; }
    void setPartTrigNr(uint32_t partTrigNrValue) { m_partTrigNr = partTrigNrValue; }

    /// get/set partition run number
    inline const uint32_t partRunNr() const { return m_partRunNr; }
    void setPartRunNr(uint32_t partRunNrValue) { m_partRunNr = partRunNrValue; }
     
    /// get/set event number since last L1 reset generated in TCS chip
    inline const uint32_t eventNr() const { return m_eventNr; } 
    void setEventNr(uint32_t eventNrValue) { m_eventNr = eventNrValue; }

    /// get/set assigned partition: bit "i" correspond to detector partition "i"
    inline const uint32_t assignedPartitions() const { return m_assignedPartitions; }
    void setAssignedPartitions(uint32_t assignedPartitionsValue) { m_assignedPartitions = assignedPartitionsValue; }

    /// get/set orbit number since start of run
    inline const uint32_t orbitNr() const { return m_orbitNr; }
    void setOrbitNr(uint32_t orbitNrValue) { m_orbitNr = orbitNrValue; }
     
        
private:
    
                               // first number in the comment represents number of bits

    uint16_t m_daqNr;          //  4: number of DAQ partition to which the L1A has been sent
    uint16_t m_triggerType;    //  4: trigger type, identical with event type in CMS header
    uint16_t m_status;         //  4: 0000 = normal rate; 1000 = low rate = warning
    uint16_t m_bxNr;           // 12: bunch cross number as counted in the TCS chip    
    uint32_t m_partTrigNr;     // 32: total number of L1A sent since start of the run
                               //     to this DAQ partition 
                               //     TODO overflow after 11.8h at 100 Hz 
//
    uint32_t m_partRunNr;      // 32: TODO clarify meaning
     
    uint32_t m_eventNr;        // 24: event number since last L1 reset generated in TCS chip 
//
    uint32_t m_assignedPartitions; // 32: bit "i" correspond to detector partition "i"
                                   //     if bit = 1, detection partition connected to actual 
                                   //     DAQ partition 
    uint32_t m_orbitNr;            // 32: orbit number since start of run
         
};

#endif /*L1GlobalTrigger_L1TcsWord_h*/
