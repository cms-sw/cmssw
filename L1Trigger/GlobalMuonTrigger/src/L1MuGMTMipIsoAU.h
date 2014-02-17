//-------------------------------------------------
//
/** \class L1MuGMTMipIsoAU
 *  L1 Global Muon Trigger MIP and ISO bit Assignment Unit.
 *
 *  It projects muons form the muon system to the calorimeter
 *  or vertex in order to assign a MIP (minimum ionizing particle) 
 *  and an ISO (isolation) bit to each muon candidate.
 *
 *  there is 1 barrel MIP & ISO bit Assignment unit and 
 *           1 endcap MIP & ISO bit Assignment unit
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
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTMipIsoAU_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTMipIsoAU_h

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
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"

class L1MuGlobalMuonTrigger;
class L1MuGMTCand;
class L1MuGMTPhiProjectionUnit;
class L1MuGMTEtaProjectionUnit;

//              ---------------------
//              -- Class Interface --
//              ---------------------


class L1MuGMTMipIsoAU {

 public:  

  /// constructor
  L1MuGMTMipIsoAU(const L1MuGlobalMuonTrigger& gmt, int id);

  /// destructor
  virtual ~L1MuGMTMipIsoAU();

  /// run GMT MIP & ISO bit assignment unit
  void run();
    
  /// clear MIP & ISO bit assignment unit
  void reset();
        
  /// print results after MIP & ISO bit assignment 
  void print() const;
    
  /// return identifier (0: barrel, 1: endcap)
  inline int id() const { return m_id; }
   
  /// return input muon (idx: 0..3: DT/CSC, 4..7: RPC)
  inline const L1MuRegionalCand* muon(int idx) const { return m_muons[idx]; } 

  /// return select matrix (idx 0..3: DT/CSC, idx 4..7: RPC)
  inline bool MIP(int idx) const { return m_MIP[idx]; }
    
  /// return select matrix (idx 0..3: DT/CSC, idx 4..7: RPC)   
  inline bool ISO(int idx) const { return m_ISO[idx]; }

  const L1MuGlobalMuonTrigger& GMT () const { return m_gmt; };
 private:

  void load();
  void assignMIP();
  void assignISO();

 private:
  const L1MuGlobalMuonTrigger& m_gmt;
  int m_id;
    
  std::vector<const L1MuRegionalCand*> m_muons;
    
  std::vector<bool> m_MIP;
  std::vector<bool> m_ISO;

  std::vector<L1MuGMTPhiProjectionUnit*> m_MIP_PPUs;
  std::vector<L1MuGMTEtaProjectionUnit*> m_MIP_EPUs;

  std::vector<L1MuGMTPhiProjectionUnit*> m_ISO_PPUs;
  std::vector<L1MuGMTEtaProjectionUnit*> m_ISO_EPUs;
};
  
#endif





