//-------------------------------------------------
//
/**  \class L1MuDTPhiLut
 *
 *   Look-up tables for phi-assignment
 *
 *
 *   $Date: 2008/04/09 15:22:31 $
 *   $Revision: 1.3 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_PHI_LUT_H
#define L1MUDT_PHI_LUT_H

//---------------
// C++ Headers --
//---------------

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


class L1MuDTPhiLut {

  public:

    /// constructor
    L1MuDTPhiLut();

    /// destructor
    virtual ~L1MuDTPhiLut();

    /// reset phi-assignment look-up tables
    void reset();
    
    /// load phi-assignment look-up tables
    int load();

    /// print phi-assignment look-up tables
    void print() const;

    /// get delta-phi for a given address (bend-angle)
    int getDeltaPhi(int idx, int address) const;

    /// get precision for look-up tables
    std::pair<unsigned short, unsigned short> getPrecision() const;

  private:

    /// set precision for look-up tables
    void setPrecision();
      
  private:

    typedef std::map<short, short, std::less<short> > LUT;

    std::vector<LUT> phi_lut;
    
    unsigned short int nbit_phi;
    unsigned short int nbit_phib;
    
};

#endif
