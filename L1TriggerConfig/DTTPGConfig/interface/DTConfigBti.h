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
#include "L1Trigger/DTUtilities/interface/DTConfig.h"

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
  
  //! Destructor 
  ~DTConfigBti();

  //! Set default parameters
  void setDefaults();  

  //! Debug flag
  inline int debug() const { return m_debug; }

  //! Max drift time in 12.5 ns steps
  inline float ST() const {
     return (float)( 0.75 * ST43() + 0.25 * RE43() ); }

  //! Max K param accepted: max bit number
  inline int  KCut() const { return m_kcut; }

  //! BTI angular acceptance in theta view                           
  inline int  KAccTheta() const { return m_kacctheta; }
  
  //! Digi-to-bti-input offset 500 (tdc units) --> wait to solve with Nicola Amapane
  inline int MCDigiOffset() const { return m_digioffset; }

  //! Minicrate "fine" sincronization parameter [0,25] ns
  inline int MCSetupTime() const { return  m_setuptime; }
  
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
  inline int PTMSflag(int patt) { return m_pattmask[patt]; }

  //! Wire mask flag 
  inline int WENflag(int wire) { return m_wiremask[wire-1]; }

  //! K left limit for left traco
  inline int LL() { return m_ll; }

  //! K right limit for left traco
  inline int LH() { return m_lh; }

  //! K left limit for center traco
  inline int CL() { return m_cl; }

  //! K right limit for center traco
  inline int CH() { return m_ch; }

  //! K left limit for right traco
  inline int RL() { return m_rl; }

  //! K right limit for right traco
  inline int RH() { return m_rh; }

  //! ST and RE parameters for drift velocity 
  inline int ST43() const { return m_4st3; }
  inline int RE43() const { return m_4re3; }

  //! Wire DEAD time parameter
  inline int DEADpar() { return m_dead;}

 
  //! Print the setup
  void print() const ;

  //! Return pointer to parameter set 
  const edm::ParameterSet* getParameterSet() { return m_ps; }


private:
  const edm::ParameterSet* m_ps;

  int m_debug;
  int m_kcut;
  int m_kacctheta;
  int m_digioffset;
  int m_setuptime;
  bool m_xon;
  int m_lts;
  int m_set;
  int m_ac1;
  int m_ac2;
  int m_ach;
  int m_acl;
  bool m_ron;
  int m_pattmask[32];
  int m_wiremask[9];
  int m_ll;
  int m_lh;
  int m_cl;
  int m_ch;
  int m_rl;
  int m_rh;
  int m_4st3;
  int m_4re3;
  int m_dead;
};

#endif
