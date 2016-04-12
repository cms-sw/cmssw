#ifndef L1MUBM_LUT_H
#define L1MUBM_LUT_H

//---------------
// C++ Headers --
//---------------

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


class L1MuBMLUTHandler {

  public:
    /// constructor
    L1MuBMLUTHandler(const L1TMuonBarrelParams &l1params);

    /// destructor
    virtual ~L1MuBMLUTHandler();

    /// print pt-assignment look-up tables
    void print_pta_lut() const;

    /// get pt-value for a given address
    int getPt(int pta_ind, int address) const;

    /// get pt-assignment LUT threshold
    int getPtLutThreshold(int pta_ind) const;

    /// print phi-assignment look-up tables
    void print_phi_lut() const;

    /// get delta-phi for a given address (bend-angle)
    int getDeltaPhi(int idx, int address) const;

    /// get precision for look-up tables
    std::pair<unsigned short, unsigned short> getPrecision() const;

    /// print extrapolation look-up tables
    void print_ext_lut() const;

    /// get low_value for a given address
    int getLow(int ext_ind, int address) const;

    /// get high_value for a given address
    int getHigh(int ext_ind, int address) const;


  private:
    const L1TMuonBarrelParams* l1tbmparams;

  public:



//max. number of Extrapolations
const int MAX_EXT = 12;

// extrapolation types
enum Extrapolation { EX12, EX13, EX14, EX21, EX23, EX24, EX34,
                     EX15, EX16, EX25, EX26, EX56 };

 // maximal number of pt assignment methods
const int MAX_PTASSMETH = 19;
const int MAX_PTASSMETHA = 12;

// pt assignment methods
enum PtAssMethod { PT12L,  PT12H,  PT13L,  PT13H,  PT14L,  PT14H,
                   PT23L,  PT23H,  PT24L,  PT24H,  PT34L,  PT34H,
                   PB12H,  PB13H,  PB14H,  PB21H,  PB23H,  PB24H, PB34H,
                   NODEF };

};

#endif
