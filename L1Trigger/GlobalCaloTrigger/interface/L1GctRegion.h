#ifndef L1GCTREGION_H_
#define L1GCTREGION_H_

#include <ostream>
#include <boost/cstdint.hpp>

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
  /// constructor
  L1GctRegion(int eta=0, int phi=0, unsigned et=0, bool mip=false, bool quiet=false, bool tauVeto=false, bool overFlow=false);

  // destructor
  ~L1GctRegion();

  // Getters
  int eta() const { return m_eta; }   ///< Get the eta number (0-21?) of the region
  int phi() const { return m_phi; }   ///< Get the phi number (0-17) of the region
  unsigned et() const { return m_et; }
  bool mip() const { return m_mip; }
  bool quiet() const { return m_quiet; }
  bool tauVeto() const { return m_tauVeto; }
  bool overFlow() const { return m_overFlow; }

  // Setters
  // should deprecate these methods where possible
  void setEta(int eta);
  void setPhi(int phi);
  void setEt(unsigned et);
  void setMip(bool mip);
  void setQuiet(bool quiet);
  void setTauVeto(bool tauVeto);
  void setOverFlow(bool overFlow);
  
  /// printing
  friend std::ostream& operator << (std::ostream& os, const L1GctRegion& reg);


private:

  /// global eta position number of the region (0-21)
  int m_eta;
  /// global phi position number of the region (0-17)
  int m_phi;

  unsigned m_et;
  bool m_mip;
  bool m_quiet;
  bool m_tauVeto;
  bool m_overFlow;

};

std::ostream& operator << (std::ostream& os, const L1GctRegion& reg);

#endif /*L1GCTREGION_H_*/
