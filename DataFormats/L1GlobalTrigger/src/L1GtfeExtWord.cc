/**
 * \class L1GtfeExtWord
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

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeExtWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"

// system include files
#include <boost/cstdint.hpp>
#include <iostream>
#include <iomanip>

// user include files
#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors

// empty constructor, all members set to zero;
L1GtfeExtWord::L1GtfeExtWord()
        : L1GtfeWord()
{

    for (int iB = 0; iB < NumberBstBlocks; ++iB) {
        m_bst[iB] = 0;
    }


}

// constructor from unpacked values;
L1GtfeExtWord::L1GtfeExtWord(
    boost::uint16_t boardIdValue,
    boost::uint16_t recordLengthValue,
    boost::uint16_t bxNrValue,
    boost::uint32_t setupVersionValue,
    boost::uint16_t activeBoardsValue,
    boost::uint32_t totalTriggerNrValue, // end of L1GtfeWord
    boost::uint16_t bstValue[NumberBstBlocks] )
        : L1GtfeWord(
            boardIdValue,
            recordLengthValue,
            bxNrValue,
            setupVersionValue,
            activeBoardsValue,
            totalTriggerNrValue
        )

{

    for (int iB = 0; iB < NumberBstBlocks; ++iB) {
        m_bst[iB] = bstValue[iB];
    }

}

// destructor
L1GtfeExtWord::~L1GtfeExtWord()
{

    // empty now

}

// equal operator
bool L1GtfeExtWord::operator==(const L1GtfeExtWord& result) const
{

    const L1GtfeWord gtfeResult = result;
    const L1GtfeWord gtfeThis = *this;

    if (gtfeThis != gtfeResult) {
        return false;
    }


    for (int iB = 0; iB < NumberBstBlocks; ++iB) {
        if(m_bst[iB] != result.m_bst[iB]) {
            return false;
        }
    }


    // all members identical
    return true;

}

// unequal operator
bool L1GtfeExtWord::operator!=(const L1GtfeExtWord& result) const
{

    return !( result == *this);

}

// methods

boost::uint64_t L1GtfeExtWord::gpsTime()
{

    boost::uint64_t gpst = 0ULL;

    for (int iB = GpsTimeFirstBlock; iB <= GpsTimeLastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - GpsTimeFirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        gpst = gpst |
               ( (static_cast<boost::uint64_t> (m_bst[iB])) << BstShift );

    }

    return gpst;
}


boost::uint32_t L1GtfeExtWord::turnCountNumber()
{

    boost::uint32_t tcn = 0;

    for (int iB = TurnCountNumberFirstBlock; iB <= TurnCountNumberLastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - TurnCountNumberFirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        tcn = tcn |
              ( (static_cast<boost::uint32_t> (m_bst[iB])) << BstShift );

    }

    return tcn;
}

boost::uint32_t L1GtfeExtWord::lhcFillNumber()
{

    boost::uint32_t lhcfn = 0;

    for (int iB = LhcFillNumberFirstBlock; iB <= LhcFillNumberLastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - LhcFillNumberFirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        lhcfn = lhcfn |
                ( (static_cast<boost::uint32_t> (m_bst[iB])) << BstShift );

    }

    return lhcfn;
}

boost::uint32_t L1GtfeExtWord::totalIntensityBeam1()
{

    boost::uint32_t tib = 0;

    for (int iB = TotalIntensityBeam1FirstBlock; iB <= TotalIntensityBeam1LastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - TotalIntensityBeam1FirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        tib = tib |
              ( (static_cast<boost::uint32_t> (m_bst[iB])) << BstShift );

    }

    return tib;
}

boost::uint32_t L1GtfeExtWord::totalIntensityBeam2()
{

    boost::uint32_t tib = 0;

    for (int iB = TotalIntensityBeam2FirstBlock; iB <= TotalIntensityBeam2LastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - TotalIntensityBeam2FirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        tib = tib |
              ( (static_cast<boost::uint32_t> (m_bst[iB])) << BstShift );

    }

    return tib;
}

boost::uint16_t L1GtfeExtWord::beamMomentum()
{

    boost::uint16_t bm = 0;

    for (int iB = BeamMomentumFirstBlock; iB <= BeamMomentumLastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - BeamMomentumFirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        bm = bm | ( m_bst[iB] << BstShift );

    }

    return bm;
}

boost::uint16_t L1GtfeExtWord::bstMasterStatus()
{

    boost::uint16_t bms = 0;

    for (int iB = BstMasterStatusFirstBlock; iB <= BstMasterStatusLastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - BstMasterStatusFirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        bms = bms | ( m_bst[iB] << BstShift );

    }

    return bms;
}

boost::uint16_t L1GtfeExtWord::machineMode()
{

    boost::uint16_t mm = 0;

    for (int iB = MachineModeFirstBlock; iB <= MachineModeLastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - MachineModeFirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        mm = mm | ( m_bst[iB] << BstShift );

    }

    return mm;
}

boost::uint16_t L1GtfeExtWord::particleTypeBeam1()
{

    boost::uint16_t ptb = 0;

    for (int iB = ParticleTypeBeam1FirstBlock; iB <= ParticleTypeBeam1LastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - ParticleTypeBeam1FirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        ptb = ptb | ( m_bst[iB] << BstShift );

    }

    return ptb;
}

boost::uint16_t L1GtfeExtWord::particleTypeBeam2()
{

    boost::uint16_t ptb = 0;

    for (int iB = ParticleTypeBeam2FirstBlock; iB <= ParticleTypeBeam2LastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - ParticleTypeBeam2FirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        ptb = ptb | ( m_bst[iB] << BstShift );

    }

    return ptb;

}

// get/set BST for block iB
const uint16_t L1GtfeExtWord::bst(int iB) const
{

    if (iB < 0 || iB > NumberBstBlocks) {
        throw cms::Exception("BstIndexError")
        << "\nError: index for BST array out of range. Allowed range: [0, "
        << NumberBstBlocks << ") " << std::endl;

    } else {
        return m_bst[iB];
    }

}

void L1GtfeExtWord::setBst(const uint16_t bstVal, const int iB)
{

    if (iB < 0 || iB > NumberBstBlocks) {
        throw cms::Exception("BstIndexError")
        << "\nError: index for BST array out of range. Allowed range: [0, "
        << NumberBstBlocks << ") " << std::endl;

    } else {
        m_bst[iB] = bstVal;
    }

}

// set the BST block for index iB from a 64-bits word
void L1GtfeExtWord::setBst(const boost::uint64_t& word64, const int iB)
{

    // keep capitalization for similarity with other functions
    const int scaledIB = iB%(sizeof(word64)*8/BstBitSize);
    const boost::uint64_t BstMask = 0x0000000000000000ULL | (BstBlockMask << scaledIB);
    const int BstShift = BstBitSize*scaledIB;

    m_bst[iB] = (word64 & BstMask) >> BstShift;

}

// set the BST block in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtfeExtWord::setBstWord64(boost::uint64_t& word64, int iB, int iWord)
{

    // keep capitalization for similarity with other functions
    const int scaledIB = iB%(sizeof(word64)*8/BstBitSize);
    const int BstShift = BstBitSize*scaledIB;
    const int BstWord = iB/(sizeof(word64)*8/BstBitSize) + BstFirstWord;

    if (iWord == BstWord) {
        word64 = word64 |
                 (static_cast<boost::uint64_t> (m_bst[iB]) << BstShift);
    }


}

// reset the content of a L1GtfeExtWord
void L1GtfeExtWord::reset()
{

    L1GtfeWord::reset();

    for (int iB = 0; iB < NumberBstBlocks; ++iB) {
        m_bst[iB] = 0;
    }

}

// pretty print the content of a L1GtfeWord
void L1GtfeExtWord::print(std::ostream& myCout) const
{

    myCout << "\n L1GtfeExtWord::print \n" << std::endl;

    int sizeW64 = 64;
    int dataBlocksPerLine = sizeW64/8; // 8x8 bits per line

    L1GtfeWord::print(myCout);

    myCout << "\n  BST ";
    for (int iB = 0; iB < NumberBstBlocks; ++iB) {

        myCout << "\n" << std::hex << " hex: ";

        for (int jB = iB; jB < dataBlocksPerLine + iB; ++jB) {
            myCout
            << std::setw(2) << std::setfill('0') << m_bst[iB]
            << "   " << std::setfill(' ');
        }

        myCout << "\n"<< std::dec << " dec: " ;

        for (int jB = iB; jB < dataBlocksPerLine + iB; ++jB) {
            myCout
            << std::setw(3) << std::setfill('0') << m_bst[iB]
            << "  " << std::setfill(' ');
        }

        myCout << std::endl;

        iB += dataBlocksPerLine;
    }

}

void L1GtfeExtWord::unpack(const unsigned char* gtfePtr)
{
    LogDebug("L1GtfeExtWord")
    << "\nUnpacking GTFE block.\n"
    << std::endl;

    L1GtfeWord::unpack(gtfePtr);

    // TODO make BlockSize protected & use friends instead of creating L1GtfeWord?
    L1GtfeWord gtfeWord;
    const unsigned char* gtfeExtPtr = gtfePtr + gtfeWord.getSize();

    const boost::uint64_t* payload =
        reinterpret_cast<boost::uint64_t*>(const_cast<unsigned char*>(gtfeExtPtr));

    if ( edm::isDebugEnabled() ) {

        for (int iWord = BstFirstWord; iWord < BlockSizeExt; ++iWord) {

            int jWord = iWord - BstFirstWord;
            LogTrace("L1GtfeExtWord")
            << std::setw(4) << iWord << "  "
            << std::hex << std::setfill('0')
            << std::setw(16) << payload[jWord]
            << std::dec << std::setfill(' ')
            << std::endl;

        }
    }

    int blocksPerWord = sizeof(boost::uint64_t)*8/BstBitSize;

    for (int iB = 0; iB < NumberBstBlocks; ++iB) {

        // keep capitalization for similarity with other functions
        int BstWord = iB/blocksPerWord;

        setBst(payload[BstWord], iB);

    }

}




// static class members
const int L1GtfeExtWord::NumberBstBlocks;

// block description in the raw GT record

// index of first word for BST blocks
const int L1GtfeExtWord::BstFirstWord = 2;

// size in bits for a BST block
const int L1GtfeExtWord::BstBitSize = 8;

// BST block mask, correlated with the number of bits of a block
// 8 bit = 0xFF
const boost::uint64_t L1GtfeExtWord::BstBlockMask = 0xFFULL;

// block size in 64bits words
const int L1GtfeExtWord::BlockSizeExt = 6;        // 6 x 64bits

// BST blocks: conversion to defined quantities (LHC-BOB-ES-0001)

const int L1GtfeExtWord::GpsTimeFirstBlock = 0;
const int L1GtfeExtWord::GpsTimeLastBlock = 7;

const int L1GtfeExtWord::TurnCountNumberFirstBlock = 8;
const int L1GtfeExtWord::TurnCountNumberLastBlock = 11;

const int L1GtfeExtWord:: LhcFillNumberFirstBlock = 12;
const int L1GtfeExtWord:: LhcFillNumberLastBlock = 15;

const int L1GtfeExtWord:: TotalIntensityBeam1FirstBlock = 16;
const int L1GtfeExtWord:: TotalIntensityBeam1LastBlock = 19;

const int L1GtfeExtWord::TotalIntensityBeam2FirstBlock = 20;
const int L1GtfeExtWord::TotalIntensityBeam2LastBlock = 23;

const int L1GtfeExtWord::BeamMomentumFirstBlock = 24;
const int L1GtfeExtWord::BeamMomentumLastBlock = 25;

const int L1GtfeExtWord::BstMasterStatusFirstBlock = 26;
const int L1GtfeExtWord::BstMasterStatusLastBlock =  26;

const int L1GtfeExtWord::MachineModeFirstBlock = 27;
const int L1GtfeExtWord::MachineModeLastBlock = 27;

const int L1GtfeExtWord::ParticleTypeBeam1FirstBlock = 28;
const int L1GtfeExtWord::ParticleTypeBeam1LastBlock = 28;

const int L1GtfeExtWord::ParticleTypeBeam2FirstBlock = 29;
const int L1GtfeExtWord::ParticleTypeBeam2LastBlock = 29;

