//-------------------------------------------------
//
/** \class L1MuGMTSorter
 *  L1 Global Muon Trigger Sorter.
*/
//
//   $Date: 2003/12/19 10:22:04 $
//   $Revision: 1.4 $
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

using namespace std;

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
    
    /// return vector with all muon candidates
    inline const vector<const L1MuGMTExtendedCand*>& Cands() const { return m_MuonCands; }

  private:

    const L1MuGlobalMuonTrigger&  m_gmt;
    vector<const L1MuGMTExtendedCand*>    m_MuonCands;

};

#endif
