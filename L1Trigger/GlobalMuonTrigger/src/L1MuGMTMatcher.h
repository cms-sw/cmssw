//-------------------------------------------------
//
/** \class L1MuGMTMatcher
 *  Matching Unit in the L1 Global Muon Trigger.
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
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTMatcher_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTMatcher_h

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

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMatrix.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"

class L1MuGlobalMuonTrigger;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuGMTMatcher {

  public:

    static const unsigned int MaxMatch = 4;

    /// constructor
    L1MuGMTMatcher(const L1MuGlobalMuonTrigger& gmt, int id);

    /// destructor
    virtual ~L1MuGMTMatcher();

    /// run Matcher
    void run();
    
    /// clear Matcher
    void reset();

    /// print matching results
    void print();
    
    /// return identifier
    inline int id() const { return m_id; } 

    /// return pair matrix
    const L1MuGMTMatrix<bool>& pairM() const { return pairMatrix; }

    /// return pair matrix
    bool pairM(int i,int j) const { return pairMatrix(i,j); }
    
  private:
    
    void load();
    
    void match();

    //    int compareEtaPhi(float eta1, float phi1, float eta2, float phi2);
    int lookup_mq(int i, int j);
    
  private:

    const L1MuGlobalMuonTrigger& m_gmt;
    int m_id;

    std::vector<const L1MuRegionalCand*> first;
    std::vector<const L1MuRegionalCand*> second;

    L1MuGMTMatrix<int>  matchQuality;
    L1MuGMTMatrix<bool> pairMatrix;
   
};
  
#endif
