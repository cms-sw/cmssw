/**\class CSCRPCDigi
 *
 * Digi for RPC data coming thru RAT-ALCT-DDU.
 *
 * $Date: 2006/03/01 09:40:21 $
 * $Revision: 1.2 $
 *
 * \author N. Terentiev, CMU
 */


#include <DataFormats/CSCDigi/interface/CSCRPCDigi.h>

#include <iostream>
#include <bitset>


/// Constructors

CSCRPCDigi::CSCRPCDigi (int rpc, int pad, int bxn, int tbin){
  rpc_ = rpc;
  pad_ = pad; 
  bxn_ = bxn;
  tbin_ = tbin;
}

/// Copy
CSCRPCDigi::CSCRPCDigi(const CSCRPCDigi& digi) {
  rpc_ = digi.getRpc();
  pad_ = digi.getPad(); 
  bxn_ = digi.getBXN();
  tbin_ = digi.getTbin();
}
/// Default
CSCRPCDigi::CSCRPCDigi (){
  rpc_ = 0;
  pad_ = 0; 
  bxn_ = 0;
  tbin_ = 0;
}


/// Assignment
CSCRPCDigi& 
CSCRPCDigi::operator=(const CSCRPCDigi& digi){
  rpc_ = digi.getRpc();
  pad_ = digi.getPad(); 
  bxn_ = digi.getBXN();
  tbin_ = digi.getTbin();
  return *this;
}

/// Debug
void CSCRPCDigi::print() const {
  std::cout << "RPC = " << getRpc()
	    << "  Pad = " << getPad()
	    << "  Tbin = " << getTbin() 
	    << "  BXN = " << getBXN() << std::endl;
}



