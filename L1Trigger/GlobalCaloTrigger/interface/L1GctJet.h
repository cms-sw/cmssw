#ifndef L1GCTJET_H_
#define L1GCTJET_H_

#include <boost/cstdint.hpp> //for uint16_t

/*
 * A GCT jet candidate
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 */

class L1GctJet
{

public:

  L1GctJet(uint16_t rank=0, uint16_t eta=0, uint16_t phi=0, bool tauVeto=true);
  ~L1GctJet();
  
  // set rank and position bits
  void setRank(uint16_t rank) { m_rank = rank; }
  void setEta(uint16_t eta) { m_eta = eta; }
  void setPhi(uint16_t phi) { m_phi = phi; }
  void setTauVeto(bool tauVeto) { m_tauVeto = tauVeto; }
  
  // get rank and position bits
  uint16_t rank()const { return m_rank; }
  uint16_t eta()const { return m_eta; }
  uint16_t phi()const { return m_phi; }
  bool tauVeto()const { return m_tauVeto; }
	
  //ostream& operator << (ostream& os, const L1GctJet& s);
  
  ///Setup an existing jet all in one go
  void setupJet(uint16_t rank, uint16_t eta, uint16_t phi, bool tauVeto=true);
  
  //! Converts a jet with local jetfinder co-ordinates (11*2) to GCT output global format
  /*! 'jetFinderPhiIndex' is the vector index of the jetfinder in the wheel card,
   *  running from 0-8. 'wheelId' is the wheelJetFPGA id number (0 or 1),
   *  to determine which eta half of CMS we are in.*/
  L1GctJet convertToGlobalJet(int jetFinderPhiIndex, int wheelId);
  
  ///start of the HF if we are using local jetfinder co-ordinates (11*2 in eta*phi)
  static const int LOCAL_ETA_HF_START = 7;
  ///External access to the rank bitwidth
  static const int RANK_BITWIDTH = 6;
  
 private:
  
  static const int ETA_BITWIDTH = 4;
  static const int PHI_BITWIDTH = 5;
  
  uint16_t m_rank;
  uint16_t m_eta;
  uint16_t m_phi;
  bool m_tauVeto;
  
};

#endif /*L1GCTJET_H_*/
