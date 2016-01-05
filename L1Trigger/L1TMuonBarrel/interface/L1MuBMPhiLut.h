//-------------------------------------------------
//
/**  \class L1MuBMPhiLut
 *
 *   Look-up tables for phi-assignment
 *
 *
 *   $Date: 2007/03/30 07:48:02 $
 *   $Revision: 1.1 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBM_PHI_LUT_H
#define L1MUBM_PHI_LUT_H

//---------------
// C++ Headers --
//---------------

#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"

#include <vector>
#include <map>
#include <utility>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


//              ---------------------
//              -- Class Interface --
//              ---------------------


class L1MuBMPhiLut {

  public:
typedef std::map<short, short, std::less<short> > LUT;

    /// constructor
    L1MuBMPhiLut(const L1TMuonBarrelParams &l1params);

    /// destructor
    virtual ~L1MuBMPhiLut();

    /// print phi-assignment look-up tables
    void print() const;

    /// get delta-phi for a given address (bend-angle)
    int getDeltaPhi(int idx, int address) const;

    /// get precision for look-up tables
    std::pair<unsigned short, unsigned short> getPrecision() const;

  private:

    const L1TMuonBarrelParams *l1tbmphiparams;


};

#endif
