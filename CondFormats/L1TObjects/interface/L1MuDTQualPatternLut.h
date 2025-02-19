//-------------------------------------------------
//
/**  \class L1MuDTQualPatternLut
 *
 *   Look-up tables for eta matching unit (EMU)
 *   stores lists of qualified patterns and
 *   coarse eta values 
 *
 *
 *   $Date: 2008/04/09 15:22:31 $
 *   $Revision: 1.2 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_QUALPATTERN_LUT_H
#define L1MUDT_QUALPATTERN_LUT_H

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

class L1MuDTQualPatternLut {

  public:

    typedef std::pair< short, short > LUTID;
    typedef std::pair< short, std::vector<short> > LUTCONT;
    typedef std::map< LUTID, LUTCONT > LUT;
    typedef LUT::iterator EMULut_iter;
    
    /// constructor
    L1MuDTQualPatternLut();

    /// destructor
    virtual ~L1MuDTQualPatternLut();

    /// reset look-up tables
    void reset();
    
    /// load look-up tables
    int load();

    /// print look-up tables
    void print() const;

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
    
  private:

    LUT m_lut;        // coarse eta values and list of qualified patterns
    
};

#endif
