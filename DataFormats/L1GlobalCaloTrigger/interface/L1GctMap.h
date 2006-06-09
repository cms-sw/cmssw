#ifndef L1GCTMAP_H
#define L1GCTMAP_H

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctRegion.h"

/*!
 * \author Jim Brooke
 * \date May 2006
 */

/*! \class L1GctMap
 * \brief Get position information from Region, jet, EM objects
 *
 * This is a singleton, which breaks coding rules. Will be converted
 * to appropriate mechanism once we know what that is!
 *
 */


class L1GctMap {
 public:

  static const unsigned N_RGN_ETA;
  static const unsigned N_RGN_PHI;

 public:
  
  /// destructor
  ~L1GctMap();

  /// return the map
  static L1GctMap* getMap() { 
    if (m_instance==0) { m_instance = new L1GctMap(); }
    return m_instance;
  }
 
  /// get the RCT crate number
  unsigned rctCrate(L1GctRegion r);

  /// get the SC number
  unsigned sourceCard(L1GctRegion r);

  /// get the eta index within an RCT crate
  unsigned rctEta(L1GctRegion r);

  /// get the phi index within an RCT crate
  unsigned rctPhi(L1GctRegion r);

  /// get the global eta index
  unsigned eta(L1GctRegion r);

  /// get the global phi index
  unsigned phi(L1GctRegion r);

  // get physical eta 
  // double eta(L1GctRegion r);

  // get physical eta 
  //  double phi(L1GctRegion r);

  /// get ID from eta, phi indices
  unsigned id(unsigned ieta, unsigned iphi);


 private:        // methods

  L1GctMap();    // constructor, this is a singleton

 private:        // data

  static L1GctMap * m_instance;  // the instance, this is a singleton


};

#endif
