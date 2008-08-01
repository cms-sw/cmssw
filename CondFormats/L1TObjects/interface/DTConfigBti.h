//-------------------------------------------------
//
/**  \class DTConfigBti
 *
 *   Configurable parameters and constants 
 *   for Level-1 Muon DT Trigger - Bti chip 
 *
 *   \author  S. Vanini
 *
 */
//
//--------------------------------------------------
#ifndef DT_CONFIG_BTI_H
#define DT_CONFIG_BTI_H

//---------------
// C++ Headers --
//---------------

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/L1TObjects/interface/DTConfig.h"
#include "CondFormats/L1TObjects/interface/BitArray.h"

#include <boost/cstdint.hpp>

//              ---------------------
//              -- Class Interface --
//              ---------------------


class DTConfigBti : DTConfig {

  public:
  
  /*  //! constants: first and last step to start trigger finding
  static const unsigned int NSTEPL=24, NSTEPF=9;
  */

  //! Constructor
  DTConfigBti(const edm::ParameterSet& ps);

  //! Empty Constructor
  DTConfigBti() {;}
   
  //! Destructor 
  ~DTConfigBti();

  //! Set default parameters
  void setDefaults(const edm::ParameterSet& ps);  

  //! Debug flag
  inline int debug() const { return m_debug; }

  //! Max drift time in 12.5 ns steps
  inline float ST() const {
     return (float)( 0.75 * ST43() + 0.25 * RE43() ); }

  //! Max K param accepted: max bit number
  inline int  KCut() const { return m_kcut; }

  //! BTI angular acceptance in theta view                           
  inline int  KAccTheta() const { return m_kacctheta; }
  
  //! X-patterns flag XON: activates tracks passing on the same side of 3 wires
  inline bool XON() const { return m_xon; }
  
  //! LTS
  inline int LTS() const { return m_lts; }
  
  //! SET
  inline int SET() const { return m_set; }
  
  //! Acceptance pattern AC1                                           
  inline int AccPattAC1() const { return m_ac1; }
  
  //! Acceptance pattern AC2                                           
  inline int  AccPattAC2() const { return m_ac2; }
  
  //! Acceptance pattern ACH                                           
  inline int  AccPattACH() const { return m_ach; }
  
  //! Acceptance pattern ACL                                           
  inline int  AccPattACL() const { return m_acl; }

  //! Redundant patterns flag RONDTBti/src/DTBtiChipEquations.cc:  
  inline bool RONflag() const { return m_ron; }

  //! Pattern mask flag 
  inline int PTMSflag(int patt) const { return m_pattmask.element(patt); }

  //! Wire mask flag 
  inline int WENflag(int wire) const { return m_wiremask.element(wire-1); }

  //! K left limit for left traco
  inline int LL() const { return m_ll; }

  //! K right limit for left traco
  inline int LH() const { return m_lh; }

  //! K left limit for center traco
  inline int CL() const { return m_cl; }

  //! K right limit for center traco
  inline int CH() const { return m_ch; }

  //! K left limit for right traco
  inline int RL() const { return m_rl; }

  //! K right limit for right traco
  inline int RH() const { return m_rh; }

  //! ST and RE parameters for drift velocity 
  inline int ST43() const { return m_4st3; }
  inline int RE43() const { return m_4re3; }

  //! Wire DEAD time parameter
  inline int DEADpar() const { return m_dead;}
 
  //! Print the setup
  void print() const ;

/*   //! Return pointer to parameter set  */
/*   const edm::ParameterSet* getParameterSet() { return m_ps; } */


private:
  //  const edm::ParameterSet* m_ps;

  unsigned short int m_debug;
  unsigned short int m_kcut;
  unsigned short int m_kacctheta;
  bool m_xon;
  unsigned short int m_lts;
  unsigned short int m_set;
  unsigned short int m_ac1;
  unsigned short int m_ac2;
  unsigned short int m_ach;
  unsigned short int m_acl;
  bool m_ron;
  BitArray<32> m_pattmask;
  BitArray<9>  m_wiremask;
  unsigned short int m_ll;
  unsigned short int m_lh;
  unsigned short int m_cl;
  unsigned short int m_ch;
  unsigned short int m_rl;
  unsigned short int m_rh;
  unsigned short int m_4st3;
  unsigned short int m_4re3;
  unsigned short int m_dead;
};

#endif
