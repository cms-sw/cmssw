#ifndef L1GCTJET_H_
#define L1GCTJET_H_

#include <boost/cstdint.hpp> //for uint16_t
#include <functional>
#include <ostream>

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

/*!
 * \author Jim Brooke & Robert Frazier
 * \date April 2006
 */

/*! \class L1GctJet
 * \brief A Level-1 jet candidate, used within GCT emulation
 * 
 *  Move this to DataFormats/L1GlobalCaloTrigger if possible
 */

class L1GctJetCand;
class L1GctJetEtCalibrationLut;

class L1GctJet
{

public:
  //Statics
  enum numberOfBits {
    kRawsumBitWidth = 10,
    kRawsumMaxValue = (1<<kRawsumBitWidth) - 1
  };
  
  //Constructors/destructors
  L1GctJet(uint16_t rawsum=0, unsigned eta=0, unsigned phi=0, bool overFlow=false, bool forwardJet=true, bool tauVeto=true);
  ~L1GctJet();
  
  // set rawsum and position bits
  void setRawsum(uint16_t rawsum) { m_rawsum = rawsum & kRawsumMaxValue; m_overFlow |= (rawsum > kRawsumMaxValue); }
  void setDetId(L1CaloRegionDetId detId) { m_id = detId; }
  void setOverFlow(bool overFlow) { m_overFlow = overFlow; }
  void setTauVeto(bool tauVeto) { m_tauVeto = tauVeto; }
  void setForward(bool forward) { m_forwardJet = forward; }
  
  // get rawsum and position bits
  uint16_t rawsum()const { return m_rawsum; }
  bool tauVeto()const { return m_tauVeto; }

  /// get overflow
  bool overFlow() const { return m_overFlow ; }

  /// test whether this jet candidate is a valid tau jet	
  bool isTauJet()     const { return (!m_forwardJet && !m_tauVeto); } 

  /// test whether this jet candidate is a (non-tau) central jet
  bool isCentralJet() const { return (!m_forwardJet && m_tauVeto); } 

  /// test whether this jet candidate is a forward jet	
  bool isForwardJet() const { return m_forwardJet; } 

  /// test whether this jet candidate has been filled	
  bool isNullJet() const { return ((m_rawsum==0) && (globalEta()==0) && (globalPhi()==0)); } 

  friend std::ostream& operator << (std::ostream& os, const L1GctJet& cand);
  
  /// test whether two jets are the same
  bool operator== (const L1GctJet& cand) const;
  
  /// test whether two jets are different
  bool operator!= (const L1GctJet& cand) const;
  
  ///Setup an existing jet all in one go
  void setupJet(uint16_t rawsum, unsigned eta, unsigned phi, bool overFlow, bool forwardJet, bool tauVeto=true);
  
  /// eta value in global CMS coordinates
  unsigned globalEta() const { return m_id.ieta(); } 

  /// phi value in global CMS coordinates
  unsigned globalPhi() const { return m_id.iphi(); } 

  /// eta value in global CMS coordinates
  unsigned rctEta() const { return m_id.rctEta(); } 

  /// phi value in global CMS coordinates
  unsigned rctPhi() const { return m_id.rctPhi(); } 

  /// eta value as encoded in hardware at the GCT output
  unsigned hwEta() const; 

  /// phi value as encoded in hardware at the GCT output
  unsigned hwPhi() const; 

  /// Function to convert from internal format to external jet candidates at the output of the jetFinder 
  L1GctJetCand jetCand(const L1GctJetEtCalibrationLut* lut) const;

  /// The two separate Lut outputs
  uint16_t rank(const L1GctJetEtCalibrationLut* lut) const;
  unsigned calibratedEt(const L1GctJetEtCalibrationLut* lut) const;


 private:

  uint16_t m_rawsum;
  /// region id, encodes eta and phi
  L1CaloRegionDetId m_id;
  bool m_overFlow;
  bool m_forwardJet;
  bool m_tauVeto;

  uint16_t lutValue (const L1GctJetEtCalibrationLut* lut) const;

};

std::ostream& operator << (std::ostream& os, const L1GctJet& cand);

#endif /*L1GCTJET_H_*/
