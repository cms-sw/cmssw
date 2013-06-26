//-------------------------------------------------
//
/** \class L1MuGMTSortRankUnit
 *
 *  L1 Global Muon Trigger Sort Rank Unit.
 * 
 *  Return sort rank based on look-up-tables.
 *
 *
*/
//
//   $Date: 2006/05/15 13:56:02 $
//   $Revision: 1.1 $
//
//   Author :
//   H. Sakulin           HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTSortRankUnit_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTSortRankUnit_h

//---------------
// C++ Headers --
//---------------

#include <vector>

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class L1MuRegionalCand;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuGMTSortRankUnit {

  public:  

    /// constructor
    L1MuGMTSortRankUnit();

    /// destructor
    virtual ~L1MuGMTSortRankUnit();
     
    /// Sort Rank Table
    static unsigned sort_rank(const L1MuRegionalCand*); 
 
    /// Very low quality bits
    static unsigned getVeryLowQualityLevel(const L1MuRegionalCand*); 

    /// Diable bit
    static bool isDisabled(const L1MuRegionalCand*); 
};
  
#endif
