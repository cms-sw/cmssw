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

// system include files
#include <iomanip>
#include <stdint.h>

// user include files
#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"


// constructors

// empty constructor, all members set to zero;
L1GtfeExtWord::L1GtfeExtWord() :
    L1GtfeWord(), m_bstSource(0) {

    // empty

}

// all members set to zero, m_bst has bstSizeBytes zero elements
L1GtfeExtWord::L1GtfeExtWord(int bstSizeBytes) :
    L1GtfeWord(), m_bstSource(0) {

    m_bst.resize(bstSizeBytes);

}


// constructor from unpacked values, m_bst size taken from bstValue
L1GtfeExtWord::L1GtfeExtWord(
        cms_uint16_t boardIdValue, cms_uint16_t recordLength1Value,
        cms_uint16_t recordLengthValue, cms_uint16_t bxNrValue,
        cms_uint32_t setupVersionValue, cms_uint16_t activeBoardsValue,
        cms_uint16_t altNrBxBoardValue, cms_uint32_t totalTriggerNrValue, // end of L1GtfeWord
        const std::vector<cms_uint16_t>& bstValue, cms_uint16_t bstSourceValue) :
    L1GtfeWord(
            boardIdValue, recordLength1Value, recordLengthValue, bxNrValue, setupVersionValue,
            activeBoardsValue, altNrBxBoardValue, totalTriggerNrValue), m_bst(bstValue),
            m_bstSource(bstSourceValue)

{

    // empty
}

// destructor
L1GtfeExtWord::~L1GtfeExtWord() {

    // empty now

}

// equal operator
bool L1GtfeExtWord::operator==(const L1GtfeExtWord& result) const
{

    // base class

    const L1GtfeWord gtfeResult = result;
    const L1GtfeWord gtfeThis = *this;

    if (gtfeThis != gtfeResult) {
        return false;
    }


    //

    for (unsigned int iB = 0; iB < m_bst.size(); ++iB) {
        if(m_bst[iB] != result.m_bst[iB]) {
            return false;
        }
    }

    if ( m_bstSource != result.m_bstSource) {
        return false;
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

const cms_uint64_t L1GtfeExtWord::gpsTime() const
{

    cms_uint64_t gpst = 0ULL;

    // return 0 if BST message too small
    int bstSize = m_bst.size();
    if (GpsTimeLastBlock >= bstSize) {
        return gpst;
    }

    for (int iB = GpsTimeFirstBlock; iB <= GpsTimeLastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - GpsTimeFirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        gpst = gpst |
               ( (static_cast<cms_uint64_t> (m_bst[iB])) << BstShift );

    }

    return gpst;
}

void L1GtfeExtWord::setGpsTime(const cms_uint64_t gpsTimeValue) {

    // return if BST message too small
    int bstSize = m_bst.size();
    if (GpsTimeLastBlock >= bstSize) {

        edm::LogError("L1GtfeExtWord") << "Error: BST message length "
            << bstSize << " smaller than the required GpsTimeLastBlock "
            << GpsTimeLastBlock << "\n Cannot set GpsTime" << std::endl;

        return;
    }

    for (int iB = GpsTimeFirstBlock; iB <= GpsTimeLastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - GpsTimeFirstBlock;
        const int BstShift = BstBitSize*scaledIB;
        const cms_uint64_t BstMask = 0x0000000000000000ULL | (BstBlockMask << BstShift);

        m_bst[iB] = static_cast<cms_uint16_t> ((gpsTimeValue & BstMask) >> BstShift);

        //LogTrace("L1GtfeExtWord")
        //<< "BstShift: value [dec] = " << BstShift << "\n"
        //<< "BstBlockMask: value [hex] = "  << std::hex << BstBlockMask << "\n"
        //<< "BstMask: value [hex] = "<< BstMask << std::dec
        //<< std::endl;

        //LogTrace("L1GtfeExtWord")
        //<< "BST block " << iB << ": value [hex] = " << std::hex << m_bst[iB] << std::dec
        //<< std::endl;

    }

}

const cms_uint16_t L1GtfeExtWord::bstMasterStatus() const
{

    cms_uint16_t bms = 0;

    // return 0 if BST message too small
    int bstSize = m_bst.size();
    if (BstMasterStatusLastBlock >= bstSize) {
        return bms;
    }

    for (int iB = BstMasterStatusFirstBlock; iB <= BstMasterStatusLastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - BstMasterStatusFirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        bms = bms | ( m_bst[iB] << BstShift );

    }

    return bms;
}


const cms_uint32_t L1GtfeExtWord::turnCountNumber() const
{

    cms_uint32_t tcn = 0;

    // return 0 if BST message too small
    int bstSize = m_bst.size();
    if (TurnCountNumberLastBlock >= bstSize) {
        return tcn;
    }

    for (int iB = TurnCountNumberFirstBlock; iB <= TurnCountNumberLastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - TurnCountNumberFirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        tcn = tcn |
              ( (static_cast<cms_uint32_t> (m_bst[iB])) << BstShift );

    }

    return tcn;
}

const cms_uint32_t L1GtfeExtWord::lhcFillNumber() const
{

    cms_uint32_t lhcfn = 0;

    // return 0 if BST message too small
    int bstSize = m_bst.size();
    if (LhcFillNumberLastBlock >= bstSize) {
        return lhcfn;
    }

    for (int iB = LhcFillNumberFirstBlock; iB <= LhcFillNumberLastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - LhcFillNumberFirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        lhcfn = lhcfn |
                ( (static_cast<cms_uint32_t> (m_bst[iB])) << BstShift );

    }

    return lhcfn;
}

const cms_uint16_t L1GtfeExtWord::beamMode() const
{

    cms_uint16_t bm = 0;

    // return 0 if BST message too small
    int bstSize = m_bst.size();
    if (BeamModeLastBlock >= bstSize) {
        return bm;
    }

    for (int iB = BeamModeFirstBlock; iB <= BeamModeLastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - BeamModeFirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        bm = bm | ( m_bst[iB] << BstShift );

    }

    return bm;
}

const cms_uint16_t L1GtfeExtWord::particleTypeBeam1() const
{

    cms_uint16_t ptb = 0;

    // return 0 if BST message too small
    int bstSize = m_bst.size();
    if (ParticleTypeBeam1LastBlock >= bstSize) {
        return ptb;
    }

    for (int iB = ParticleTypeBeam1FirstBlock; iB <= ParticleTypeBeam1LastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - ParticleTypeBeam1FirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        ptb = ptb | ( m_bst[iB] << BstShift );

    }

    return ptb;
}

const cms_uint16_t L1GtfeExtWord::particleTypeBeam2() const
{

    cms_uint16_t ptb = 0;

    // return 0 if BST message too small
    int bstSize = m_bst.size();
    if (ParticleTypeBeam2LastBlock >= bstSize) {
        return ptb;
    }

    for (int iB = ParticleTypeBeam2FirstBlock; iB <= ParticleTypeBeam2LastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - ParticleTypeBeam2FirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        ptb = ptb | ( m_bst[iB] << BstShift );

    }

    return ptb;

}

const cms_uint16_t L1GtfeExtWord::beamMomentum() const
{

    cms_uint16_t bm = 0;

    // return 0 if BST message too small
    int bstSize = m_bst.size();
    if (BeamMomentumLastBlock >= bstSize) {
        return bm;
    }

    for (int iB = BeamMomentumFirstBlock; iB <= BeamMomentumLastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - BeamMomentumFirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        bm = bm | ( m_bst[iB] << BstShift );

    }

    return bm;
}

const cms_uint32_t L1GtfeExtWord::totalIntensityBeam1() const
{

    cms_uint32_t tib = 0;

    // return 0 if BST message too small
    int bstSize = m_bst.size();
    if (TotalIntensityBeam1LastBlock >= bstSize) {
        return tib;
    }

    for (int iB = TotalIntensityBeam1FirstBlock; iB <= TotalIntensityBeam1LastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - TotalIntensityBeam1FirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        tib = tib |
              ( (static_cast<cms_uint32_t> (m_bst[iB])) << BstShift );

    }

    return tib;
}

const cms_uint32_t L1GtfeExtWord::totalIntensityBeam2() const
{

    cms_uint32_t tib = 0;

    // return 0 if BST message too small
    int bstSize = m_bst.size();
    if (TotalIntensityBeam2LastBlock >= bstSize) {
        return tib;
    }

    for (int iB = TotalIntensityBeam2FirstBlock; iB <= TotalIntensityBeam2LastBlock; ++iB) {

        // keep capitalization for similarity with other functions
        const int scaledIB = iB - TotalIntensityBeam2FirstBlock;
        const int BstShift = BstBitSize*scaledIB;

        tib = tib |
              ( (static_cast<cms_uint32_t> (m_bst[iB])) << BstShift );

    }

    return tib;
}


// get/set BST for block iB
const uint16_t L1GtfeExtWord::bst(int iB) const
{

    int NumberBstBlocks = m_bst.size();

    if (iB < 0 || iB >= NumberBstBlocks) {
        throw cms::Exception("BstIndexError")
        << "\nError: index for BST array out of range. Allowed range: [0, "
        << NumberBstBlocks << ") " << std::endl;

    } else {
        return m_bst[iB];
    }

}

void L1GtfeExtWord::setBst(const uint16_t bstVal, const int iB)
{

    int NumberBstBlocks = m_bst.size();

    if (iB < 0 || iB >= NumberBstBlocks) {
        throw cms::Exception("BstIndexError")
        << "\nError: index for BST array out of range. Allowed range: [0, "
        << NumberBstBlocks << ") " << std::endl;

    } else {
        m_bst[iB] = bstVal;
    }

}

// set the BST block for index iB from a 64-bits word
void L1GtfeExtWord::setBst(const cms_uint64_t& word64, const int iB)
{

    // keep capitalization for similarity with other functions //FIXME check it again
    const int scaledIB = iB%(sizeof(word64)*8/BstBitSize);
    const int BstShift = BstBitSize*scaledIB;
    const cms_uint64_t BstMask = 0x0000000000000000ULL | (BstBlockMask << BstShift);

    m_bst[iB] = static_cast<cms_uint16_t> ((word64 & BstMask) >> BstShift);

}

// set the BST block in a 64-bits word, having the index iWord
// in the GTFE raw record
void L1GtfeExtWord::setBstWord64(cms_uint64_t& word64, int iB, int iWord)
{

    // keep capitalization for similarity with other functions
    const int scaledIB = iB%(sizeof(word64)*8/BstBitSize);
    const int BstShift = BstBitSize*scaledIB;
    const int BstWord = iB/(sizeof(word64)*8/BstBitSize) + BstFirstWord;

    if (iWord == BstWord) {
        word64 = word64 |
                 (static_cast<cms_uint64_t> (m_bst[iB]) << BstShift);
    }


}


// set the hex message indicating the source of BST message from a 64-bits word
void L1GtfeExtWord::setBstSource(const cms_uint64_t& word64) {

    m_bstSource = (word64 & BstSourceMask) >> BstSourceShift;

}

// set hex message indicating the source of BST message in a 64-bits word,
// having the index iWord in the GTFE raw record
void L1GtfeExtWord::setBstSourceWord64(cms_uint64_t& word64, const int iWord) {

    // BST always in the last word of GTFE extended - size must be correct!
    int gtfeSize = this->getSize();

    int BstSourceWord = gtfeSize/8 - 1; // counting starts at 0

    if (iWord == BstSourceWord) {
        word64 = word64 | (static_cast<cms_uint64_t> (m_bstSource)
                           << BstSourceShift);
    }

}


// get the size of the GTFE block in GT EVM record (in multiple of 8 bits)
const unsigned int L1GtfeExtWord::getSize() const {

    L1GtfeWord gtfeWord;
    unsigned int gtfeSize = gtfeWord.getSize();

    unsigned int gtfeExtSize;

    // 2 bytes to write if real BST message or simulated BST message
    unsigned int bytesBstWriter = 2;

    // size of BST block, using rounded 64-bit words (8 bytes per 64-bit word)

    unsigned int bstSize = m_bst.size();

    if ( (bstSize +bytesBstWriter )%8 == 0) {
        gtfeExtSize = gtfeSize + bstSize + bytesBstWriter;
    }
    else {
        gtfeExtSize = gtfeSize + bstSize + bytesBstWriter + (8 - (bstSize + bytesBstWriter)%8 );
    }

    return gtfeExtSize;
}



// resize the BST vector to get the right size of the block
void L1GtfeExtWord::resize(int bstSizeBytes) {

    m_bst.resize(bstSizeBytes);

}

// reset the content of a L1GtfeExtWord
void L1GtfeExtWord::reset()
{

    L1GtfeWord::reset();
    m_bst.clear();

}

// pretty print the content of a L1GtfeWord
void L1GtfeExtWord::print(std::ostream& myCout) const {

    myCout << "\n L1GtfeExtWord::print \n" << std::endl;

    unsigned int sizeW64 = 64;
    unsigned int dataBlocksPerLine = sizeW64 / 8; // 8x8 bits per line

    L1GtfeWord::print(myCout);

    unsigned int numberBstBlocks = m_bst.size();

    myCout << "\n  BST ";

    if (numberBstBlocks == 0) {

        myCout << "\n  BST source [hex]: " << std::hex << std::setw(4)
                << std::setfill('0') << m_bstSource << std::setfill(' ')
                << std::dec << std::endl;

        return;
    }

    for (unsigned int iB = 0; iB < numberBstBlocks; iB += dataBlocksPerLine) {

        myCout << "\n" << std::hex << " hex: ";

        for (unsigned int jB = iB; jB < dataBlocksPerLine + iB; ++jB) {

            if (jB >= numberBstBlocks) {
                break;
            }

            myCout << std::setw(2) << std::setfill('0') << m_bst[jB] << "   "
                    << std::setfill(' ');
        }

        myCout << "\n" << std::dec << " dec: ";

        for (unsigned int jB = iB; jB < dataBlocksPerLine + iB; ++jB) {

            if (jB >= numberBstBlocks) {
                break;
            }

            myCout << std::setw(3) << std::setfill('0') << m_bst[jB] << "  "
                    << std::setfill(' ');
        }

        myCout << std::endl;

    }

    myCout << "\n  BST source [hex]: " << std::hex << std::setw(4)
            << std::setfill('0') << m_bstSource << std::setfill(' ')
            << std::dec << std::endl;

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

    const cms_uint64_t* payload =
        reinterpret_cast<cms_uint64_t*>(const_cast<unsigned char*>(gtfeExtPtr));

    int BlockSizeExt = this->getSize()/8;
    int NumberBstBlocks = m_bst.size();

    if (edm::isDebugEnabled() ) {

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

    int blocksPerWord = sizeof(cms_uint64_t)*8/BstBitSize;

    for (int iB = 0; iB < NumberBstBlocks; ++iB) {

        // keep capitalization for similarity with other functions
        int BstWord = iB/blocksPerWord;

        setBst(payload[BstWord], iB);

    }

}




// static class members

// block description in the raw GT record

// index of first word for BST blocks
const int L1GtfeExtWord::BstFirstWord = 2;

// size in bits for a BST block
const int L1GtfeExtWord::BstBitSize = 8;

// BST block mask, correlated with the number of bits of a block
// 8 bit = 0xFF
const cms_uint64_t L1GtfeExtWord::BstBlockMask = 0xFFULL;

// BST blocks: conversion to defined quantities (LHC-BOB-ES-0001)

const int L1GtfeExtWord::GpsTimeFirstBlock = 0;
const int L1GtfeExtWord::GpsTimeLastBlock = 7;

const int L1GtfeExtWord::BstMasterStatusFirstBlock = 17;
const int L1GtfeExtWord::BstMasterStatusLastBlock =  17;

const int L1GtfeExtWord::TurnCountNumberFirstBlock = 18;
const int L1GtfeExtWord::TurnCountNumberLastBlock = 21;

const int L1GtfeExtWord::LhcFillNumberFirstBlock = 22;
const int L1GtfeExtWord::LhcFillNumberLastBlock = 25;

const int L1GtfeExtWord::BeamModeFirstBlock = 26;
const int L1GtfeExtWord::BeamModeLastBlock = 27;

const int L1GtfeExtWord::ParticleTypeBeam1FirstBlock = 28;
const int L1GtfeExtWord::ParticleTypeBeam1LastBlock = 28;

const int L1GtfeExtWord::ParticleTypeBeam2FirstBlock = 29;
const int L1GtfeExtWord::ParticleTypeBeam2LastBlock = 29;

const int L1GtfeExtWord::BeamMomentumFirstBlock = 30;
const int L1GtfeExtWord::BeamMomentumLastBlock = 31;

const int L1GtfeExtWord::TotalIntensityBeam1FirstBlock = 32;
const int L1GtfeExtWord::TotalIntensityBeam1LastBlock = 35;

const int L1GtfeExtWord::TotalIntensityBeam2FirstBlock = 36;
const int L1GtfeExtWord::TotalIntensityBeam2LastBlock = 39;

// BST
const cms_uint64_t L1GtfeExtWord::BstSourceMask = 0xFFFF000000000000ULL;
const int L1GtfeExtWord::BstSourceShift = 48;



