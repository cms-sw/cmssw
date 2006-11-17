#ifndef CSCRPCDigi_CSCRPCDigi_h
#define CSCRPCDigi_CSCRPCDigi_h

/**\class CSCRPCDigi
 *
 * Digi for RPC data coming thru RAT-ALCT-DDU. 
 *
 * $Date: 2006/05/16 15:22:57 $
 * $Revision: 1.5 $
 *
 * \author N. Terentiev, CMU
 */

#include <boost/cstdint.hpp>


class CSCRPCDigi{

public:

  /// Constructors
  CSCRPCDigi (int rpc, int pad, int bxn , int tbin);  /// from the rpc#, pad#, bxn#, tbin#
  CSCRPCDigi ();                             /// default

  
  /// get RPC
  int getRpc() const {return rpc_;}
  /// return pad number
  int getPad() const {return pad_;}
  /// return tbin number
  int getTbin() const {return tbin_;}
  /// return BXN
  int getBXN() const {return bxn_;}
  
  /// Print content of digi
  void print() const;


private:

  uint16_t rpc_;
  uint16_t pad_;
  uint16_t bxn_;
  uint16_t tbin_;


};




#include<iostream>
// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCRPCDigi& digi) {
  return o << " RPC = " << digi.getRpc() << "  Pad = "<< digi.getPad()
	   << "  Tbin = " << digi.getTbin() << "  Bxn = " << digi.getBXN();
}



#endif
