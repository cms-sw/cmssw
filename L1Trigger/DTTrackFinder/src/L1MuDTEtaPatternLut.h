//-------------------------------------------------
//
/**  \class L1MuDTEtaPatternLut
 *
 *   Look-up table for eta track finder
 *
 *
 *   $Date: 2006/06/01 00:00:00 $
 *   $Revision: 1.1 $
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

class L1MuDTEtaPattern;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTEtaPatternLut {

  public:

    typedef map<int, L1MuDTEtaPattern*, less<int> > LUT;
    typedef LUT::iterator ETFLut_iter;
    
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
    L1MuDTEtaPattern* getPattern(int id) const;
    
    /// return number of entries in the LUT
    inline int size() const { return m_lut.size(); }

    /// return iterator which points to the first entry of the LUT
    inline ETFLut_iter begin() { return m_lut.begin(); }

    /// return iterator which points to the one-past-last entry of the LUT
    inline ETFLut_iter end() { return m_lut.end(); }

  private:

    LUT m_lut;
    
};

#endif
