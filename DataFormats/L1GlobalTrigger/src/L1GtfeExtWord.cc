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

// TODO FIXME write the methods to retrieve the quantities from BST
// now they are dummy

boost::uint64_t L1GtfeExtWord::gpsTime()
{

    boost::uint64_t gpst = 0ULL;
    return gpst;

}


boost::uint32_t L1GtfeExtWord::turnCountNumber()
{

    boost::uint32_t tcn = 0;
    return tcn;

}

boost::uint32_t L1GtfeExtWord::lhcFillNumber()
{

    boost::uint32_t lhcfn = 0;
    return lhcfn;

}

boost::uint32_t L1GtfeExtWord::totalIntensityBeam1()
{

    boost::uint32_t tib = 0;
    return tib;

}

boost::uint32_t L1GtfeExtWord::totalIntensityBeam2()
{

    boost::uint32_t tib = 0;
    return tib;

}

boost::uint16_t L1GtfeExtWord::beamMomentum()
{

    boost::uint16_t bm = 0;
    return bm;

}

boost::uint16_t L1GtfeExtWord::bstMasterStatus()
{

    boost::uint32_t tcn = 0;
    return tcn;

}

boost::uint16_t L1GtfeExtWord::machineMode()
{

    boost::uint16_t mm = 0;
    return mm;

}

boost::uint16_t L1GtfeExtWord::particleTypeBeam1()
{

    boost::uint16_t ptb = 0;
    return ptb;

}

boost::uint16_t L1GtfeExtWord::particleTypeBeam2()
{

    boost::uint16_t ptb = 0;
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
void L1GtfeExtWord::setBst(uint16_t bstVal, int iB)
{

    if (iB < 0 || iB > NumberBstBlocks) {
        throw cms::Exception("BstIndexError")
        << "\nError: index for BST array out of range. Allowed range: [0, "
        << NumberBstBlocks << ") " << std::endl;

    } else {
        m_bst[iB] = bstVal;
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

// static class members
const int L1GtfeExtWord::NumberBstBlocks;
