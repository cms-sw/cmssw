//-------------------------------------------------
//
/**  \class L1MuDTQualPatternLut
 *
 *   Look-up tables for eta matching unit (EMU)
 *   stores lists of qualified patterns and
 *   coarse eta values 
 *
 *
 *   $Date: 2007/03/30 07:48:02 $
 *   $Revision: 1.1 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBMT_QUALPATTERN_LUT_H
#define L1MUBMT_QUALPATTERN_LUT_H

//---------------
// C++ Headers --
//---------------

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/L1TObjects/interface/L1TriggerLutFile.h"

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

class L1MuBMTQualPatternLut {

  public:

    typedef std::pair< short, short > LUTID;
    typedef std::pair< short, std::vector<short> > LUTCONT;
    typedef std::map< LUTID, LUTCONT > LUT;
    typedef LUT::iterator EMULut_iter;
    
    /// constructor
    L1MuBMTQualPatternLut();

    /// destructor
    virtual ~L1MuBMTQualPatternLut();

    /// reset look-up tables
    void reset();
    
    /// load look-up tables
    int load();

    /// print look-up tables
    void print() const;

    int getIgnoredLines(L1TriggerLutFile file) const;

    /// get coarse eta value for a given sector processor [1-6] and address [1-22]
    int getCoarseEta(int sp, int adr) const;
 
    /// get list of qualified patterns for a given sector processor [1-6] and address [1-22]
    const std::vector<short>& getQualifiedPatterns(int sp, int adr) const;

    /// return number of entries in the LUT
    inline int size() const { return m_lut.size(); }

    /// return iterator which points to the first entry of the LUT
    inline EMULut_iter begin() { return m_lut.begin(); }

    /// return iterator which points to the one-past-last entry of the LUT
    inline EMULut_iter end() { return m_lut.end(); }
    
  public:

    LUT m_lut;        // coarse eta values and list of qualified patterns
    

 COND_SERIALIZABLE;
};

#endif
