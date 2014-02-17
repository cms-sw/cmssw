//-------------------------------------------------
//
/** \class L1MuGMTMerger
 *
 *  L1 Global Muon Trigger Merger.
 * 
 *  There are two GMT Mergers. One for the barrel (id=0) and 
 *  one for the forward (id=1) part.
 *
 *  The merger receives four DT(CSC) muons and four RPC muons.
 *  Each DT(CSC) muon is either merged with an RPC one or
 *  passed through.
 *
 *  To simplify the C++ model, the merger conatians also the conversion
 *  units and sort rank units as well as the MergeMethodSelection unit 
 *  which are separate units in the hardware.
 *  
 *
 *
*/
//
//   $Date: 2007/03/23 18:51:35 $
//   $Revision: 1.2 $
//
//   Author :
//   H. Sakulin           HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTMerger_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTMerger_h

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

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMatrix.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"

class L1MuGlobalMuonTrigger;


//              ---------------------
//              -- Class Interface --
//              ---------------------


class L1MuGMTMerger {

  public:  

    /// constructor
    L1MuGMTMerger(const L1MuGlobalMuonTrigger& gmt, int id);

    /// destructor
    virtual ~L1MuGMTMerger();

    /// run GMT Merger
    void run();
    
    /// clear Merger
    void reset();
        
    /// print results after selection 
    void print() const;
    
    /// return identifier
    inline int id() const { return m_id; }
   
    /// return std::vector with all muon candidates
    inline const std::vector<L1MuGMTExtendedCand*>& Cands() const { return m_MuonCands; }
     
   private:

    void load();
    void merge();

    void createMergedCand(int idx_dtcsc, int idx_rpc);
    void createDTCSCCand(int idx_dtcsc);
    void createRPCCand(int idx_rpc);

    int doSpecialMerge(unsigned MMconfig) const;
    int doANDMerge(unsigned MMconfig) const;
    int selectDTCSC(unsigned MMconfig, int by_rank, int by_pt, int by_combi) const;

    unsigned convertedEta(const L1MuRegionalCand* mu) const;
    unsigned projectedPhi(const L1MuRegionalCand* mu) const;
    unsigned sysign(const L1MuRegionalCand* mu) const;
 
    /// Merge Rank Table
    int merge_rank(const L1MuRegionalCand*) const; 
    
  private:

    const L1MuGlobalMuonTrigger& m_gmt;
    int m_id;
    
    std::vector<const L1MuRegionalCand*> dtcsc_mu; 
    std::vector<const L1MuRegionalCand*> rpc_mu;
    
    std::vector<L1MuGMTExtendedCand*> m_MuonCands; // up to eight

    std::vector<int> singleRank;              //@ 8 bits
};
  
#endif
