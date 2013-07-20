//-------------------------------------------------
//
/** \class L1MuGMTSorter
 *  L1 Global Muon Trigger Sorter.
*/
//
//   $Date: 2007/03/23 18:51:35 $
//   $Revision: 1.2 $
//
//   Author :
//   N. Neumeister            CERN EP
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTSorter_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTSorter_h

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

class L1MuGlobalMuonTrigger;
class L1MuGMTExtendedCand;

//              ---------------------
//              -- Class Interface --
//              ---------------------


class L1MuGMTSorter {

  public:

    /// constructor
    L1MuGMTSorter(const L1MuGlobalMuonTrigger&);

    /// destructor
    virtual ~L1MuGMTSorter();

    /// run Sorter
    virtual void run();
    
    /// reset Sorter
    virtual void reset();
    
    /// print results after sorting
    void print();

    /// return number of found muon candidates after sorter
    inline int numberOfCands() const { return m_MuonCands.size(); }
    
    /// return std::vector with all muon candidates
    inline const std::vector<const L1MuGMTExtendedCand*>& Cands() const { return m_MuonCands; }

  private:

    const L1MuGlobalMuonTrigger&  m_gmt;
    std::vector<const L1MuGMTExtendedCand*>    m_MuonCands;

};

#endif
