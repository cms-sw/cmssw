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
#include <vector>
#include <iosfwd>

// user include files
#include "FWCore/Utilities/interface/typedefs.h"

// base class
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"

// forward declarations

// class declaration
class L1GtfeExtWord : public L1GtfeWord
{

public:

    /// constructors
    L1GtfeExtWord();    // empty constructor, all members set to zero;

    /// all members set to zero, m_bst has bstSizeBytes zero elements
    L1GtfeExtWord(int bstSizeBytes);

    /// constructor from unpacked values, m_bst size taken from bstValue
    L1GtfeExtWord(
        cms_uint16_t boardIdValue,
        cms_uint16_t recordLengthValue,
        cms_uint16_t recordLength1Value,
        cms_uint16_t bxNrValue,
        cms_uint32_t setupVersionValue,
        cms_uint16_t activeBoardsValue,
        cms_uint16_t altNrBxBoardValue,
        cms_uint32_t totalTriggerNrValue, // end of L1GtfeWord
        const std::vector<cms_uint16_t>& bstValue,
        cms_uint16_t bstSourceValue
    );


    /// destructor
    virtual ~L1GtfeExtWord();

    /// equal operator
    bool operator==(const L1GtfeExtWord&) const;

    /// unequal operator
    bool operator!=(const L1GtfeExtWord&) const;

public:

    /// get the full BST block
    inline const std::vector<cms_uint16_t>&  bst() const {
        return m_bst;
    }

    /// get the size of the BST block
    inline const unsigned int bstLengthBytes() const {
        return m_bst.size();
    }

public:

    /// LHC-BOB-ES-0001 (EDMS 638899)

    const cms_uint64_t gpsTime() const;
    void setGpsTime(const cms_uint64_t);

    const cms_uint16_t bstMasterStatus() const;
    const cms_uint32_t turnCountNumber() const;
    const cms_uint32_t lhcFillNumber() const;
    const cms_uint16_t beamMode() const;
    const cms_uint16_t particleTypeBeam1() const;
    const cms_uint16_t particleTypeBeam2() const;
    const cms_uint16_t beamMomentum() const;
    const cms_uint32_t totalIntensityBeam1() const;
    const cms_uint32_t totalIntensityBeam2() const;



public:

    /// get/set BST block for index iB
    const cms_uint16_t bst(int iB) const;
    void setBst(const cms_uint16_t bstVal, const int iB);

    /// set the BST block for index iB from a 64-bits word
    void setBst(const cms_uint64_t& word64, const int iB);

    /// set the BST block in a 64-bits word, having the index iWord
    /// in the GTFE raw record
    void setBstWord64(cms_uint64_t& word64, int iB, const int iWord);


    /// get/set hex message indicating the source of BST message
    inline const cms_uint16_t bstSource() const {
        return m_bstSource;
    }

    inline void setBstSource(const cms_uint16_t bstSourceVal) {
        m_bstSource = bstSourceVal;
    }

    /// set the hex message indicating the source of BST message from a 64-bits word
    void setBstSource(const cms_uint64_t& word64);

    /// set hex message indicating the source of BST message in a 64-bits word,
    /// having the index iWord in the GTFE raw record
    void setBstSourceWord64(cms_uint64_t& word64, const int iWord);

public:

    /// get the size of the GTFE block in GT EVM record (in multiple of 8 bits)
    const unsigned int getSize() const;

public:

    /// resize the BST vector to get the right size of the block
    void resize(int bstSizeBytes);

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
    static const cms_uint64_t BstBlockMask;

    /// BST blocks: conversion to defined quantities (LHC-BOB-ES-0001)

    static const int GpsTimeFirstBlock;
    static const int GpsTimeLastBlock;

    static const int BstMasterStatusFirstBlock;
    static const int BstMasterStatusLastBlock;

    static const int TurnCountNumberFirstBlock;
    static const int TurnCountNumberLastBlock;

    static const int LhcFillNumberFirstBlock;
    static const int LhcFillNumberLastBlock;

    static const int BeamModeFirstBlock;
    static const int BeamModeLastBlock;

    static const int ParticleTypeBeam1FirstBlock;
    static const int ParticleTypeBeam1LastBlock;

    static const int ParticleTypeBeam2FirstBlock;
    static const int ParticleTypeBeam2LastBlock;

    static const int BeamMomentumFirstBlock;
    static const int BeamMomentumLastBlock;

    static const int TotalIntensityBeam1FirstBlock;
    static const int TotalIntensityBeam1LastBlock;

    static const int TotalIntensityBeam2FirstBlock;
    static const int TotalIntensityBeam2LastBlock;

    // BST message source written always in the last word of GTFE extended
    static const cms_uint64_t BstSourceMask;

    static const int BstSourceShift;



private:

    /// BST message - each byte is an vector element
    std::vector<cms_uint16_t> m_bst;

    /// hex message indicating the source of BST message (beam or simulated)
    cms_uint16_t m_bstSource;

};

#endif /*L1GlobalTrigger_L1GtfeExtWord_h*/
