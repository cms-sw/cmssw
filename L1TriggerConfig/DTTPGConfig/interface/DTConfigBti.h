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
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfig.h"
#include "L1TriggerConfig/DTTPGConfig/interface/BitArray.h"
#include <cstdint>

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
  DTConfigBti() { ; }

  //! Constructor from string
  DTConfigBti(int debug, unsigned short int* buffer);

  //! Destructor
  ~DTConfigBti() override;

  //! Set default parameters
  void setDefaults(const edm::ParameterSet& ps);

  //! Debug flag
  inline int debug() const { return m_debug; }

  //! Max drift time in 12.5 ns steps
  inline float ST() const { return (float)(0.75 * ST43() + 0.25 * RE43()); }

  //! Max K param accepted: max bit number
  inline int KCut() const { return m_kcut; }

  //! BTI angular acceptance in theta view
  inline int KAccTheta() const { return m_kacctheta; }

  //! X-patterns flag XON: activates tracks passing on the same side of 3 wires
  inline bool XON() const { return m_xon; }

  //! LTS
  inline int LTS() const { return m_lts; }

  //! SET
  inline int SET() const { return m_set; }

  //! Acceptance pattern AC1
  inline int AccPattAC1() const { return m_ac1; }

  //! Acceptance pattern AC2
  inline int AccPattAC2() const { return m_ac2; }

  //! Acceptance pattern ACH
  inline int AccPattACH() const { return m_ach; }

  //! Acceptance pattern ACL
  inline int AccPattACL() const { return m_acl; }

  //! Redundant patterns flag RONDTBti/src/DTBtiChipEquations.cc:
  inline bool RONflag() const { return m_ron; }

  //! Pattern mask flag
  inline int PTMSflag(int patt) const { return m_pattmask.element(patt); }

  //! Wire mask flag
  inline int WENflag(int wire) const { return m_wiremask.element(wire - 1); }

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
  inline int DEADpar() const { return m_dead; }

  //! Set single parameter functions
  //! Set debug flag
  inline void setDebug(int debug) { m_debug = debug; }

  //! Set Max K param accepted: max bit number
  inline void setKCut(int KCut) { m_kcut = KCut; }

  //! Set BTI angular acceptance in theta view
  inline void setKAccTheta(int KAccTh) { m_kacctheta = KAccTh; }

  //! Set X-patterns flag XON: activates tracks passing on the same side of 3 wires
  inline void setXON(bool XON) { m_xon = XON; }

  //! Set LTS
  inline void setLTS(int LTS) { m_lts = LTS; }

  //! Set SET
  inline void setSET(int SET) { m_set = SET; }

  //! Set Acceptance pattern AC1
  inline void setAccPattAC1(int AC1) { m_ac1 = AC1; }

  //! Set Acceptance pattern AC2
  inline void setAccPattAC2(int AC2) { m_ac2 = AC2; }

  //! Set Acceptance pattern ACH
  inline void setAccPattACH(int ACH) { m_ach = ACH; }

  //! Set Acceptance pattern ACL
  inline void setAccPattACL(int ACL) { m_acl = ACL; }

  //! Set Redundant patterns flag RONDTBti/src/DTBtiChipEquations.cc:
  inline void setRONflag(bool RON) { m_ron = RON; }

  //! Set Pattern mask flag
  inline void setPTMSflag(int mask, int patt) { m_pattmask.set(patt, mask); }

  //! Set Wire mask flag
  inline void setWENflag(int mask, int wire) { m_wiremask.set(wire - 1, mask); }

  //! Set K left limit for left traco
  inline void setLL(int LL) { m_ll = LL; }

  //! Set K right limit for left traco
  inline void setLH(int LH) { m_lh = LH; }

  //! Set K left limit for center traco
  inline void setCL(int CL) { m_cl = CL; }

  //! Set K right limit for center traco
  inline void setCH(int CH) { m_ch = CH; }

  //! Set K left limit for right traco
  inline void setRL(int RL) { m_rl = RL; }

  //! Set K right limit for right traco
  inline void setRH(int RH) { m_rh = RH; }

  //! Set ST and RE parameters for drift velocity
  inline void setST43(int ST43) { m_4st3 = ST43; }
  inline void setRE43(int RE43) { m_4re3 = RE43; }

  //! Wire DEAD time parameter
  inline void setDEADpar(int DEAD) { m_dead = DEAD; }

  //! Print the setup
  void print() const;

  /*   //! Return pointer to parameter set  */
  /*   const edm::ParameterSet* getParameterSet() { return m_ps; } */

private:
  //  const edm::ParameterSet* m_ps;

  int8_t m_debug;
  int8_t m_kcut;
  int8_t m_kacctheta;
  bool m_xon;
  int8_t m_lts;
  int8_t m_set;
  int8_t m_ac1;
  int8_t m_ac2;
  int8_t m_ach;
  int8_t m_acl;
  bool m_ron;
  BitArray<32> m_pattmask;
  BitArray<9> m_wiremask;
  int8_t m_ll;
  int8_t m_lh;
  int8_t m_cl;
  int8_t m_ch;
  int8_t m_rl;
  int8_t m_rh;
  int8_t m_4st3;
  int8_t m_4re3;
  int8_t m_dead;
};

#endif
