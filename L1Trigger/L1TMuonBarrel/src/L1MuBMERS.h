//-------------------------------------------------
//
/**  \class L1MuBMERS
 *
 *   Extrapolation Result Selector (Quality Sorter Unit):
 *
 *   selects the 2 best (highest target quality)
 *   out of the 12 (6) extrapolations
 *   performed per start track segment
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBM_ERS_H
#define L1MUBM_ERS_H

//---------------
// C++ Headers --
//---------------

#include <utility>

//----------------------
// Base Class Headers --
//----------------------

#include "L1Trigger/L1TMuonBarrel/interface/L1AbstractProcessor.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

class L1MuBMTrackSegPhi;
class L1MuBMSEU;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuBMERS : public L1AbstractProcessor {

  public:

    /// constructor
    L1MuBMERS(const L1MuBMSEU& );

    /// destructor
    ~L1MuBMERS() override;

    /// run L1MuBMERS
    void run() override;

    /// reset ERS
    void reset() override;

    /// return extrapolation quality
    inline unsigned int quality(int id) const { return m_quality[id]; }

    /// return extrapolation address; (address = 15 indicates negative ext. result)
    inline unsigned short int address(int id) const { return m_address[id]; }

    /// return pointer to start and target track segment
    std::pair<const L1MuBMTrackSegPhi*, const L1MuBMTrackSegPhi*> ts(int id) const;

  private:

    const L1MuBMSEU& m_seu;

    unsigned short int m_quality[2];   //@@ 1 bit
    unsigned short int m_address[2];   //@@ 4 bits

    const L1MuBMTrackSegPhi* m_start[2];
    const L1MuBMTrackSegPhi* m_target[2];

};

#endif
