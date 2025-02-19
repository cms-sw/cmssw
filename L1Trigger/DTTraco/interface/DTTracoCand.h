//-------------------------------------------------
//
/**  \class DTTracoCand
 *
 *   Implementation of DTTracoChip candidate 
 * 
 *
 *   $Date: 2008/06/30 13:42:21 $
 *   $Revision: 1.3 $
 *
 *   \author C. Grandi, S. Vanini
 *
 *   Modifications: 
 *   S.V. store BtiTrig pointer instead of TrigData
 */
//
//--------------------------------------------------
#ifndef DT_TRACO_CAND_H
#define DT_TRACO_CAND_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTBtiTrigData;
class DTTracoChip;

//----------------------
// Base Class Headers --
//----------------------
#include "L1TriggerConfig/DTTPGConfig/interface/BitArray.h"

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTTracoCand {

  public:

    /// Constructor
    DTTracoCand() {}

    /// Constructor
    DTTracoCand(DTTracoChip* tc, const DTBtiTrigData* btitr, int pos, int step);

    /// Constructor
    DTTracoCand(const DTTracoCand& tccand);

    /// Assignment operator
    DTTracoCand& operator=(const DTTracoCand& tccand);

    /// Destructor 
    ~DTTracoCand();

    /// set candidate used (unusable)
    inline void setUsed() { _usable=0; }

    /// set candidate unused (usable)
    inline void setUnused() {_usable=1; }

    /// set quality bits for first/second track
    void setBits(int itk);

    /// Return parent TRACO
    inline DTTracoChip* Traco() const { return _traco; }

    /// Return associated BTI trigger
    inline const DTBtiTrigData* BtiTrig() const { return _btitr; }

    /// Return Bunch crossing
    inline int step() const { return _step; }

    /// Return position inside TRACO
    inline int position() const { return _position; }

    /// Return K-KRAD
    inline int K() const { return _tcK; }

    /// Return local X coordinate
    inline int X() const { return _tcX; }

    /// Check if candidate is usable
    inline int usable() const { return _usable; }

    /// returns true if it has smaller K w.r.t. DTTracoChip center K (sort ascend)
    bool operator < ( const DTTracoCand& c) const { return _tcK<c._tcK; }
    // bool operator < ( const DTTracoCand& c) const { return _dataword<c._dataword; }

    /*
    /// returns true if it has smaller K w.r.t. DTTracoChip center K (sort ascend)
    inline bool closer ( const DTTracoCand& cand1, const DTTracoCand& cand2) const {
       return cand1<cand2; 
    }

    /// return true if it has larger K w.r.t. DTTracoChip center K (sort descend)
    inline bool wider ( const DTTracoCand& cand1, const DTTracoCand& cand2) const { 
      return cand2<cand1; 
    }
    */

    /// Print candidate
    void print() const ;

  private:

    DTTracoChip* _traco;         // Parent DTTracoChip

    const DTBtiTrigData* _btitr; // Associated BTI trigger

    // Other variables
    BitArray<7> _dataword; // the word on which sorting is done
    int _step;
    int _position;
    int _usable;
    int _tcX;
    int _tcK;

};

#endif
