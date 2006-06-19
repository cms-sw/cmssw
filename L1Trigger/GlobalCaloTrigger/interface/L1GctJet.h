#ifndef L1GCTJETCAND_H_
#define L1GCTJETCAND_H_

#include <boost/cstdint.hpp> //for uint16_t
#include <functional>
#include <ostream>

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctDigis.h"

/*!
 * \author Jim Brooke & Robert Frazier
 * \date April 2006
 */

/*! \class L1GctJet
 * \brief A Level-1 jet candidate, used within GCT emulation
 * 
 *  Move this to DataFormats/L1GlobalCaloTrigger if possible
 */

class L1GctJet
{

public:
  //Statics
  static const int LOCAL_ETA_HF_START; ///< start of the HF in 'local' jetfinder co-ordinates (11*2 in eta*phi)
  
  //Constructors/destructors
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

  /// test whether this jet candidate has been filled	
  bool isNullJet() const { return ((m_rank==0) && (m_eta==0) && (m_phi==0)); } 

  friend std::ostream& operator << (std::ostream& os, const L1GctJet& cand);
  
  ///Setup an existing jet all in one go
  void setupJet(uint16_t rank, uint16_t eta, uint16_t phi, bool tauVeto=true);
  
  //! Converts a jet with local jetfinder co-ordinates (11*2) to GCT output global format
  /*! 'jetFinderPhiIndex' is the vector index of the jetfinder in the wheel card,
   *  running from 0-8. 'wheelId' is the wheelJetFPGA id number (0 or 1),
   *  to determine which eta half of CMS we are in.*/
  L1GctJet convertToGlobalJet(int jetFinderPhiIndex, int wheelId);
  
  // comparison operator for sorting jets in the Wheel Fpga, JetFinder, and JetFinalStage
  struct rankGreaterThan : public std::binary_function<L1GctJet, L1GctJet, bool> 
  {
    bool operator()(const L1GctJet& x, const L1GctJet& y) { return x.rank() > y.rank(); }
  };
  
  /// produce a GCT jet digi
  L1GctJetCand makeJetCand();


 private:

  //Declare statics
  static const int RANK_BITWIDTH;  
  static const int ETA_BITWIDTH;
  static const int PHI_BITWIDTH;
  
  uint16_t m_rank;
  uint16_t m_eta;
  uint16_t m_phi;
  bool m_tauVeto;
  
};

std::ostream& operator << (std::ostream& os, const L1GctJet& cand);

#endif /*L1GCTJETCAND_H_*/
