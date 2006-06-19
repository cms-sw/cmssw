#ifndef L1GCTETSCALES_H
#define L1GCTETSCALES_H

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctDigis.h"

/*!
 * \author Jim Brooke
 * \date May 2006
 */

/*! \class L1GctEtScales
 * \brief Convert digital Et scales to physical units
 *
 * This is a singleton, which breaks coding rules. Will be converted
 * to appropriate mechanism once we know what that is!
 *
 */

class L1GctEtScales {
 public:
  
  /// destructor
  ~L1GctEtScales();

  /// return the map
  static L1GctEtScales* getEtScales() { 
    if (m_instance==0) { m_instance = new L1GctEtScales(); }
    return m_instance;
  }
  

  /// get electron Et
  double et(L1GctEmCand cand);

  /// get jet Et
  double et(L1GctJetCand cand);


 private:        // methods

  L1GctEtScales();    // constructor, this is a singleton

 private:        // data

  static L1GctEtScales * m_instance;  // the instance, this is a singleton


};

#endif
