//-------------------------------------------------
//
/** \class L1MuGMTEtaProjectionUnit
 *  L1 Global Muon Trigger Eta projection unit.
 *
 *  Projects a muon from the muon system to the 
 *  calorimeter or vertex and selects one or more
 *  calorimeter regions in eta to be checked for 
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
#ifndef L1TriggerGlobalMuonTrigger_L1MuGMTEtaProjectionUnit_h
#define L1TriggerGlobalMuonTrigger_L1MuGMTEtaProjectionUnit_h

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


class L1MuGMTEtaProjectionUnit {

 public:  

  /// constructor
  L1MuGMTEtaProjectionUnit(const L1MuGMTMipIsoAU& miau, int id);

  /// destructor
  virtual ~L1MuGMTEtaProjectionUnit();

  /// run eta projection unit
  void run();
    
  /// clear eta projection unit
  void reset();
        
  /// print results after eta projection
  void print() const;
    
  /// return identifier
  inline int id() const { return m_id; }
    	
  /// return eta select bit (idx: 0..13)
  inline bool isSelected(int idx) const { return m_eta_select[idx]; }

 private:

  void load();

 private:
  typedef std::bitset<14> TEtaBits;

  const L1MuGMTMipIsoAU& m_MIAU;
    
  // index: (0..31: 16*isFWD + 8*isISO + 4* isRPC + nr )
  int m_id; 
    
  const L1MuRegionalCand* m_mu;

  int m_ieta;   // region index of central region
  float m_feta; // fine grain eta inside central region
    
  TEtaBits m_eta_select;
};
  
#endif
