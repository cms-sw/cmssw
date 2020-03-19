//-------------------------------------------------
//
/**  \class L1MuDTEtaPatternLut
 *
 *   Look-up table for eta track finder
 *
 *
 *   $Date: 2007/03/30 07:48:02 $
 *   $Revision: 1.1 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBMT_ETAPATTERN_LUT_H
#define L1MUBMT_ETAPATTERN_LUT_H

//---------------
// C++ Headers --
//---------------

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "CondFormats/L1TObjects/interface/L1MuDTEtaPattern.h"
#include "CondFormats/L1TObjects/interface/L1TriggerLutFile.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuBMTEtaPatternLut {
public:
  typedef std::map<short, L1MuDTEtaPattern, std::less<short> > LUT;
  typedef LUT::const_iterator ETFLut_iter;
  typedef LUT::iterator ETFLut_Iter;

  /// constructor
  L1MuBMTEtaPatternLut();

  /// destructor
  virtual ~L1MuBMTEtaPatternLut();

  /// reset pattern look-up table
  void reset();

  /// load pattern look-up table
  int load();

  /// print pattern look-up table
  void print() const;

  int getIgnoredLines(L1TriggerLutFile file) const;

  /// get pattern with a given ID
  L1MuDTEtaPattern getPattern(int id) const;

  /// return number of entries in the LUT
  inline int size() const { return m_lut.size(); }

  /// return iterator which points to the first entry of the LUT
  inline ETFLut_iter begin() const { return m_lut.begin(); }

  /// return iterator which points to the one-past-last entry of the LUT
  inline ETFLut_iter end() const { return m_lut.end(); }

public:
  LUT m_lut;

  COND_SERIALIZABLE;
};

#endif
