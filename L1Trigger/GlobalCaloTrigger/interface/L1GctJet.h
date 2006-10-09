#ifndef L1GCTJET_H_
#define L1GCTJET_H_

#include <boost/cstdint.hpp> //for uint16_t
#include <functional>
#include <ostream>

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"

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
  static const unsigned RAWSUM_BITWIDTH;  
  
  //Constructors/destructors
  L1GctJet(uint16_t rawsum=0, unsigned eta=0, unsigned phi=0, bool tauVeto=true,
	   L1GctJetEtCalibrationLut* lut=0);
  ~L1GctJet();
  
  // set rawsum and position bits
  void setRawsum(uint16_t rawsum) { m_rawsum = rawsum; }
  void setDetId(L1CaloRegionDetId detId) { m_id = detId; }
  void setTauVeto(bool tauVeto) { m_tauVeto = tauVeto; }
  void setLut(L1GctJetEtCalibrationLut* lut) {m_jetEtCalibrationLut = lut; }
  
  // get rawsum and position bits
  uint16_t rawsum()const { return m_rawsum; }
  bool tauVeto()const { return m_tauVeto; }
  L1GctJetEtCalibrationLut* lut() const { return m_jetEtCalibrationLut; }

  uint16_t rank()      const;
  uint16_t calibratedEt() const;

  /// get overflow
  bool overFlow() const { return (m_rawsum>=(1<<RAWSUM_BITWIDTH)); }

  /// test whether this jet candidate is a valid tau jet	
  bool isTauJet()     const { return (!m_id.isForward() && !m_tauVeto); } 

  /// test whether this jet candidate is a (non-tau) central jet
  bool isCentralJet() const { return (!m_id.isForward() && m_tauVeto); } 

  /// test whether this jet candidate is a forward jet	
  bool isForwardJet() const { return m_id.isForward(); } 

  /// test whether this jet candidate has been filled	
  bool isNullJet() const { return ((m_rawsum==0) && (globalEta()==0) && (globalPhi()==0)); } 

  friend std::ostream& operator << (std::ostream& os, const L1GctJet& cand);
  
  /// test whether two jets are the same
  bool operator== (const L1GctJet& cand) const;
  
  /// test whether two jets are different
  bool operator!= (const L1GctJet& cand) const;
  
  ///Setup an existing jet all in one go
  void setupJet(uint16_t rawsum, unsigned eta, unsigned phi, bool tauVeto=true);
  
  // comparison operator for sorting jets in the Wheel Fpga, JetFinder, and JetFinalStage
  struct rankGreaterThan : public std::binary_function<L1GctJet, L1GctJet, bool> 
  {
    bool operator()(const L1GctJet& x, const L1GctJet& y) {
      return ( x.rank() > y.rank() ) ;
    }
  };
  
  /// produce a GCT jet digi
  L1GctJetCand makeJetCand();
  
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


 private:

  uint16_t m_rawsum;
  /// region id, encodes eta and phi
  L1CaloRegionDetId m_id;
  bool m_tauVeto;

  L1GctJetEtCalibrationLut* m_jetEtCalibrationLut;
  
};

std::ostream& operator << (std::ostream& os, const L1GctJet& cand);

#endif /*L1GCTJET_H_*/
