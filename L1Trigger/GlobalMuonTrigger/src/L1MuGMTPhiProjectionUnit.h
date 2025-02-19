//-------------------------------------------------
//

/** \class L1MuGMTPhiProjectionUnit
 *  L1 Global Muon Trigger Phi projection unit.
 *
 *  Projects a muon from the muon system to the 
 *  calorimeter or vertex and selects one or more
 *  calorimeter regions in phi to be checked for 
 *  MIP and Isolation.
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
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTPhiProjectionUnit_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTPhiProjectionUnit_h

//---------------
// C++ Headers --
//---------------

#include <vector>
#include <bitset>

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMatrix.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"

class L1MuGMTMipIsoAU;
class L1MuGMTCand;


//              ---------------------
//              -- Class Interface --
//              ---------------------


class L1MuGMTPhiProjectionUnit {

 public:  

  /// constructor
  L1MuGMTPhiProjectionUnit(const L1MuGMTMipIsoAU& miau, int id);

  /// destructor
  virtual ~L1MuGMTPhiProjectionUnit();

  /// run phi projection unit
  void run();
    
  /// clear phi projection unit
  void reset();
        
  /// print results after phi projection
  void print() const;
    
  /// return identifier
  inline int id() const { return m_id; }
    	
  /// return phi select bit (idx: 0..17)
  inline bool isSelected(int idx) const { return m_phi_select[idx]; }

 private:

  void load();

 private:
  typedef std::bitset<18> TPhiBits;

  const L1MuGMTMipIsoAU& m_MIAU;
    
  // index: (0..31: 16*isFWD + 8*isISO + 4* isRPC + nr )
  int m_id; 
    
  const L1MuRegionalCand* m_mu;

  int m_iphi;   // region index of central region
  float m_fphi; // fine grain phi inside central region
    
  TPhiBits m_phi_select;
};
  
#endif
