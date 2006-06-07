#ifndef L1GCTETSCALES_H
#define L1GCTETSCALES_H

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctDigis.h"

class L1GctEtScales {
 public:
  
  /// destructor
  ~L1GctEtScales();

  /// return the map
  static L1GctEtScales* getEtScales() { 
    if (m_instance==0) { m_instance = new L1GctEtScales(); }
    return m_instance;
  }
  

  /// electron Et
  double et(L1GctEmCand cand);

  /// jet Et
  double et(L1GctCenJet cand);

  /// jet Et
  double et(L1GctTauJet cand);

  /// jet Et
  double et(L1GctForJet cand);


 private:        // methods

  L1GctEtScales();    // constructor, this is a singleton

 private:        // data

  static L1GctEtScales * m_instance;  // the instance, this is a singleton


};

#endif
