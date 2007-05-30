#ifndef L1GlobalTrigger_L1GtfeExtWord_h
#define L1GlobalTrigger_L1GtfeExtWord_h

/**
 * \class L1GtfeExtWord
 * 
 * 
 * Description: L1 Global Trigger - extended GTFE block in the readout record.  
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

// base class
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

// forward declarations

// class declaration
class L1GtfeExtWord : public L1GtfeWord
{

public:

    /// number of BST blocks - one block has 8 bits
    static const int NumberBstBlocks = 30;

public:

    /// constructors
    L1GtfeExtWord();    // empty constructor, all members set to zero;

    /// constructor from unpacked values;
    L1GtfeExtWord(
        boost::uint16_t boardIdValue,
        boost::uint16_t recordLengthValue,
        boost::uint16_t bxNrValue,
        boost::uint32_t setupVersionValue,
        boost::uint16_t activeBoardsValue,
        boost::uint32_t totalTriggerNrValue, // end of L1GtfeWord
        boost::uint16_t bstValue[NumberBstBlocks]
    );


    /// destructor
    virtual ~L1GtfeExtWord();

    /// equal operator
    bool operator==(const L1GtfeExtWord&) const;

    /// unequal operator
    bool operator!=(const L1GtfeExtWord&) const;

public:

    /// get/set BST block for index iB
    const uint16_t bst(int iB) const;
    void setBst(uint16_t bstVal, int iB);

public:

    /// LHC-BOB-ES-0001 (EDMS 638899)
    boost::uint64_t gpsTime();
    boost::uint32_t turnCountNumber();
    boost::uint32_t lhcFillNumber();
    boost::uint32_t totalIntensityBeam1();
    boost::uint32_t totalIntensityBeam2();
    boost::uint16_t beamMomentum();
    boost::uint16_t bstMasterStatus();
    boost::uint16_t machineMode();
    boost::uint16_t particleTypeBeam1();
    boost::uint16_t particleTypeBeam2();


public:

    /// get the size of the GTFE block in GT EVM record (in multiple of 8 bits)
    inline const unsigned int getSize() const
    {
        int unitLengthBits = L1GlobalTriggerReadoutSetup::UnitLength;

        return BlockSizeExt*unitLengthBits;
    }

public:

    /// reset the content of a L1GtfeExtWord
    void reset();

    /// pretty print the content of a L1GtfeExtWord
    virtual void print(std::ostream& myCout) const;


private:

    /// block description in the raw GT record

    static const int TurnCountNumberWord = 0;
    static const boost::uint64_t TurnCountNumberMask = 0x00000000FFFFFFFFULL;

    // shifts could be computed from masks...
    static const int TurnCountNumberShift = 0;

    /// block size in 64bits words
    static const int BlockSizeExt = 6;        // 6 x 64bits



private:

    ///
    uint16_t m_bst[NumberBstBlocks];



};

#endif /*L1GlobalTrigger_L1GtfeExtWord_h*/
