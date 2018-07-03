//-------------------------------------------------
//
/**  \class L1MuBMTrackAssembler
 *
 *   Track Assembler:
 *
 *   The Track Assembler gets the
 *   18 Bitmap tables from the
 *   Quality Sorter Unit and links the
 *   corresponding track segments
 *   to full tracks
 *
 *   (this version corresponds to the VHDL
 *   model b_sts_7 version 7 )
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBM_TRACK_ASSEBMLER_H
#define L1MUBM_TRACK_ASSEBMLER_H

//---------------
// C++ Headers --
//---------------

#include <bitset>

//----------------------
// Base Class Headers --
//----------------------

#include "L1Trigger/L1TMuonBarrel/interface/L1AbstractProcessor.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMTrackAssParam.h"
#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMAddressArray.h"
class L1MuBMSectorProcessor;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuBMTrackAssembler : public L1AbstractProcessor {

  public:

    /// constructor
    L1MuBMTrackAssembler(const L1MuBMSectorProcessor& );

    /// destructor
    ~L1MuBMTrackAssembler() override;

    /// run Track Assembler
    void run() override;

    /// reset Track Assembler
    void reset() override;

    /// print result of Track Assembler
    void print() const;

    /// return Track Class of found track
    inline TrackClass trackClass(int id) const { return m_theTCs[id]; }

    /// return bitmap of found track
    inline const std::bitset<4>& trackBitMap(int id) const { return m_theBitMaps[id]; }

    /// is it a valid Track Class?
    inline bool isEmpty(int id) const { return (m_theTCs[id] == UNDEF); }

    /// get address of a single station of selected track candidate
    inline int address(int id, int stat) const { return m_theAddresses[id].station(stat); }

    /// get address-array of selected track candidate
    inline L1MuBMAddressArray address(int id) const { return m_theAddresses[id]; }

  private:

    /// run the first Priority Encoder Sub-Unit
    void runEncoderSubUnit1(unsigned& global, unsigned& group, unsigned& priority);

    /// run the second Priority Encoder Sub-Unit
    void runEncoderSubUnit2(unsigned& global, unsigned& group, unsigned& priority);

    /// run the first Address Assignment Sub-Unit
    void runAddressAssignment1(int global, int group);

    /// run the second Address Assignment Sub-Unit
    void runAddressAssignment2(int global, int group);

    /// 12 bit priority encoder
    static unsigned int priorityEncoder12(const std::bitset<12>& input);

    /// 4 bit priority encoder
    static unsigned int priorityEncoder4(const std::bitset<4>& input);

    /// 22 bit priority encoder
    static unsigned int priorityEncoder22(const std::bitset<22>& input);

    /// 21 bit priority encoder
    static unsigned int priorityEncoder21(const std::bitset<21>& input);

    /// 12 bit address encoder
    static unsigned int addressEncoder12(const std::bitset<12>& input);

    /// special 12 bit address encoder
    static unsigned int addressEncoder12s(const std::bitset<12>& input);

    /// get sub-bitmap of a 68-bit word
    static unsigned long subBitset68(const std::bitset<68>& input, int pos, int length);

    /// get sub-bitmap of a 56-bit word
    static unsigned long subBitset56(const std::bitset<56>& input, int pos, int length);

    /// cancel Out Table
    static std::bitset<56> getCancelationTable(unsigned int);

  private:

    const L1MuBMSectorProcessor& m_sp;

    std::bitset<68>              m_thePriorityTable1;
    std::bitset<56>              m_thePriorityTable2;
    unsigned int                 m_theLastAddress[68];
    unsigned int                 m_theLastAddressI[12];

    TrackClass                   m_theTCs[2];        // Track Classes of the 2 candidates
    std::bitset<4>               m_theBitMaps[2];    // bitmaps of Track Class
    L1MuBMAddressArray           m_theAddresses[2];  // relative addresses of the 2 candidates

};

#endif
