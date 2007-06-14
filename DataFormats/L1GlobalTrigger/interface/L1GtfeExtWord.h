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
    void setBst(const uint16_t bstVal, const int iB);

    /// set the BST block for index iB from a 64-bits word
    void setBst(const boost::uint64_t& word64, const int iB);

    /// set the BST block in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setBstWord64(boost::uint64_t& word64, int iB, int iWord);

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

    /// unpack GTFE
    /// gtfePtr pointer to the beginning of the GTFE block in the raw data
    virtual void unpack(const unsigned char* gtfePtr);

private:

    /// block description in the raw GT record

    /// index of first word for BST blocks
    static const int BstFirstWord;

    /// size in bits for a BST block
    static const int BstBitSize;

    /// BST block mask, correlated with the number of bits of a block
    /// 8 bit = 0xFF
    static const boost::uint64_t BstBlockMask;

    /// block size in 64bits words
    static const int BlockSizeExt;

    /// BST blocks: conversion to defined quantities (LHC-BOB-ES-0001)

    static const int GpsTimeFirstBlock;
    static const int GpsTimeLastBlock;

    static const int TurnCountNumberFirstBlock;
    static const int TurnCountNumberLastBlock;

    static const int  LhcFillNumberFirstBlock;
    static const int  LhcFillNumberLastBlock;

    static const int  TotalIntensityBeam1FirstBlock;
    static const int  TotalIntensityBeam1LastBlock;

    static const int TotalIntensityBeam2FirstBlock;
    static const int TotalIntensityBeam2LastBlock;

    static const int BeamMomentumFirstBlock;
    static const int BeamMomentumLastBlock;

    static const int BstMasterStatusFirstBlock;
    static const int BstMasterStatusLastBlock;

    static const int MachineModeFirstBlock;
    static const int MachineModeLastBlock;

    static const int ParticleTypeBeam1FirstBlock;
    static const int ParticleTypeBeam1LastBlock;

    static const int ParticleTypeBeam2FirstBlock;
    static const int ParticleTypeBeam2LastBlock;

private:

    ///
    uint16_t m_bst[NumberBstBlocks];


};

#endif /*L1GlobalTrigger_L1GtfeExtWord_h*/
