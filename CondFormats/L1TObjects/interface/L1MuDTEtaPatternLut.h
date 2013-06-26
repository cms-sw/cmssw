//-------------------------------------------------
//
/**  \class L1MuDTEtaPatternLut
 *
 *   Look-up table for eta track finder
 *
 *
 *   $Date: 2008/04/09 15:22:31 $
 *   $Revision: 1.3 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_ETAPATTERN_LUT_H
#define L1MUDT_ETAPATTERN_LUT_H

//---------------
// C++ Headers --
//---------------

#include <map>

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "CondFormats/L1TObjects/interface/L1MuDTEtaPattern.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTEtaPatternLut {

  public:

    typedef std::map<short, L1MuDTEtaPattern, std::less<short> > LUT;
    typedef LUT::const_iterator ETFLut_iter;
    typedef LUT::iterator       ETFLut_Iter;
    
    /// constructor
    L1MuDTEtaPatternLut();

    /// destructor
    virtual ~L1MuDTEtaPatternLut();

    /// reset pattern look-up table
    void reset();
    
    /// load pattern look-up table
    int load();

    /// print pattern look-up table
    void print() const;

    /// get pattern with a given ID
    L1MuDTEtaPattern getPattern(int id) const;
    
    /// return number of entries in the LUT
    inline int size() const { return m_lut.size(); }

    /// return iterator which points to the first entry of the LUT
    inline ETFLut_iter begin() const { return m_lut.begin(); }

    /// return iterator which points to the one-past-last entry of the LUT
    inline ETFLut_iter end() const { return m_lut.end(); }

  private:

    LUT m_lut;
    
};

#endif
