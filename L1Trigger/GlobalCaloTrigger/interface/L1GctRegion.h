#ifndef L1GCTREGION_H_
#define L1GCTREGION_H_

#include <bitset>

/*
 * A calorimeter trigger region
 * as represented in the GCT
 * author: Jim Brooke, Robert Frazier
 * date: 20/2/2006
 * 
 */

class L1GctRegion
{
public:
  L1GctRegion(unsigned long data=0);
  L1GctRegion(int eta, int phi, unsigned long et, bool mip, bool quiet, bool tauVeto, bool overFlow);
  ~L1GctRegion();

  friend std::ostream& operator << (std::ostream& os, const L1GctRegion& reg);

  // Getters
  int eta() const { return m_eta; }   ///< Get the eta number (0-21?) of the region
  int phi() const { return m_phi; }   ///< Get the phi number (0-17) of the region
  unsigned long getEt() const { return m_et.to_ulong(); }
  bool getMip() const { return m_mip; }
  bool getQuiet() const { return m_quiet; }
  bool getTauVeto() const { return m_tauVeto; }
  bool getOverFlow() const { return m_overFlow; }

  // Setters
  void setEta(int eta) { m_eta = eta; }
  void setPhi(int phi) { m_phi = phi; }
  void setEt(unsigned long et) { /*assert(et < (1 << ET_BITWIDTH));*/ m_et = et; } 
  void setMip(bool mip) { m_mip = mip; }
  void setQuiet(bool quiet) { m_quiet = quiet; }
  void setTauVeto(bool tauVeto) { m_tauVeto = tauVeto; }
  void setOverFlow(bool overFlow) { m_overFlow = overFlow; }


private:
  //Declare statics
  static const int ET_BITWIDTH = 10;

  /// global eta position number of the region (0-21)
  int m_eta;
  /// global phi position number of the region (0-17)
  int m_phi;

  std::bitset<ET_BITWIDTH> m_et;
  bool m_mip;
  bool m_quiet;
  bool m_tauVeto;
  bool m_overFlow;

};

std::ostream& operator << (std::ostream& os, const L1GctRegion& reg);

#endif /*L1GCTREGION_H_*/
