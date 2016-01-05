//-------------------------------------------------
//
/**  \class L1MuBMPtaLut
 *
 *   Look-up tables for pt-assignment
 *
 *
 *   $Date: 2007/03/30 07:48:02 $
 *   $Revision: 1.1 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBM_PTA_LUT_H
#define L1MUBM_PTA_LUT_H

//---------------
// C++ Headers --
//---------------

//#include "CondFormats/Serialization/interface/Serializable.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMTrack.h"
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"

#include <vector>
#include <map>

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


//              ---------------------
//              -- Class Interface --
//              ---------------------


class L1MuBMPtaLut {

  public:
    typedef std::map<short, short, std::less<short> > LUT;
    /// constructor
    L1MuBMPtaLut(const L1TMuonBarrelParams &l1params);

    /// destructor
    virtual ~L1MuBMPtaLut();

    /// print pt-assignment look-up tables
    void print() const;

    /// get pt-value for a given address
    int getPt(int pta_ind, int address) const;

    /// get pt-assignment LUT threshold
    int getPtLutThreshold(int pta_ind) const;

  private:
    const L1TMuonBarrelParams* l1tbmparams;



};

#endif
