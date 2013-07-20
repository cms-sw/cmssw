//-------------------------------------------------
//
/** \class L1MuGMTCancelOutUnit
 *
 *  L1 Global Muon Trigger Cancel-Out Unit.
 *
 * It compares (attempts to match) muons in the overlap 
 * region in order to cancel out duplicates. The following 
 * four instances exist:
 *
 *                       INPUTS
 * idx   unit    where   mu1   mu2   
 * ---------------------------------------
 *  0   DT/CSC   brl     DT    CSC   
 *  1   CSC/DT   fwd     CSC   DT    
 *  2   bRPC/CSC brl     bRPC  CSC   
 *  3   fRPC/DT  fwd     fRPC  DT    
 *                      (mine)  (other chip)  
 *
 * The first input muons are direct inputs to the chip where the
 * COU is located, while the second muons are CSC or DT muon from 
 * the other Logic FPGA. 
 * 
 * Along with the input muons to be compared the COU accesses information
 * on whether the input muons are matched with a cnadiate from the 
 * complementary system in the respective main matcher.
 * 
 * The output are cancel-bits which indicate whether to 
 * delete one of the inputs.
 *
*/
//
//   $Date: 2007/03/23 18:51:35 $
//   $Revision: 1.2 $
//
//   Author :
//   H. Sakulin            CERN EP 
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTCancelOutUnit_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTCancelOutUnit_h

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

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMatcher.h"

class L1MuGlobalMuonTrigger;

//              ---------------------
//              -- Class Interface --
//              ---------------------


class L1MuGMTCancelOutUnit {

  public:
    /// constructor
    L1MuGMTCancelOutUnit(const L1MuGlobalMuonTrigger& gmt, int id);

    /// destructor
    virtual ~L1MuGMTCancelOutUnit();

    /// run cancel-out unit
    void run();
    
    /// clear cancel-out unit
    void reset();

    /// print cancel-out bits
    void print();
    
    /// return ID (0..DT/CSC in barrel chip, 1..DT/CSC in fwd chip, 2..CSC/bRPC, 3..DT/fRPC)
    inline int id() const { return m_id; } 

    /// return cancel bit for DT (m_id==0 || m_id==3) or CSC (m_id==1 || m_id==2) muon 
    const bool cancelMyChipMuon(int idx) const { return m_MyChipCancelbits[idx]; }

    /// return cancel bit for barrel RPC (m_id==2) or forward RPC (m_id==3) muon
    const bool cancelOtherChipMuon(int idx) const { return m_OtherChipCancelbits[idx]; }
     
  private:
    void decide();
   
  private:

    const L1MuGlobalMuonTrigger& m_gmt;
    int m_id;

    L1MuGMTMatcher m_matcher;
    std::vector<bool> m_MyChipCancelbits;
    std::vector<bool> m_OtherChipCancelbits;
};
  
#endif
