/**\class CSCRPCDigi
 *
 * Digi for RPC data coming thru RAT-ALCT-DDU.
 *
 * $Date: 2008/10/29 18:34:41 $
 * $Revision: 1.7 $
 *
 * \author N. Terentiev, CMU
 */


#include "DataFormats/CSCDigi/interface/CSCRPCDigi.h"
#include <iostream>

/// Constructors

CSCRPCDigi::CSCRPCDigi (int rpc, int pad, int bxn, int tbin){
  rpc_ = rpc;
  pad_ = pad; 
  bxn_ = bxn;
  tbin_ = tbin;
}

/// Default
CSCRPCDigi::CSCRPCDigi (){
  rpc_ = 0;
  pad_ = 0; 
  bxn_ = 0;
  tbin_ = 0;
}

/// Debug
void CSCRPCDigi::print() const {
  std::cout << "RPC = " << getRpc()
	    << "  Pad = " << getPad()
	    << "  Tbin = " << getTbin() 
	    << "  BXN = " << getBXN() << std::endl;
}

std::ostream & operator<<(std::ostream & o, const CSCRPCDigi& digi) {
  return o << " RPC = " << digi.getRpc() << "  Pad = "<< digi.getPad()
           << "  Tbin = " << digi.getTbin() << "  Bxn = " << digi.getBXN();
}


