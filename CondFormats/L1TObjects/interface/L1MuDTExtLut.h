//-------------------------------------------------
//
/**  \class L1MuDTExtLut
 *
 *   Look-up tables for extrapolation
 *
 *
 *   $Date: 2008/04/09 15:34:54 $
 *   $Revision: 1.5 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_EXT_LUT_H
#define L1MUDT_EXT_LUT_H

//---------------
// C++ Headers --
//---------------

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


class L1MuDTExtLut {

  public:

    /// helper class for look-up tables
    class LUT {
      public:
        typedef std::map<short, short, std::less<short> > LUTmap;

        LUTmap low;
        LUTmap high;
    };

    /// constructor
    L1MuDTExtLut();

    /// destructor
    virtual ~L1MuDTExtLut();

    /// reset extrapolation look-up tables
    void reset();
    
    /// load extrapolation look-up tables
    int load();

    /// print extrapolation look-up tables
    void print() const;

    /// get low_value for a given address
    int getLow(int ext_ind, int address) const;
    
    /// get high_value for a given address
    int getHigh(int ext_ind, int address) const;

  private:

    /// set precision for look-up tables
    void setPrecision();
    
  private:

    std::vector<LUT> ext_lut;

    unsigned short int nbit_phi;
    unsigned short int nbit_phib;
    
};

#endif
