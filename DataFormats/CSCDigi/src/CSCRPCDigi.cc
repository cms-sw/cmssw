/**\class CSCRPCDigi
 *
 * Digi for RPC data coming thru RAT-ALCT-DDU.
 *
 * $Date: 2005/11/02 23:28:59 $
 * $Revision: 1.1 $
 *
 * \author N. Terentiev, CMU
 */

#include <DataFormats/CSCDigi/interface/CSCRPCDigi.h>

#include <iostream>
#include <bitset>


/// Constructors

CSCRPCDigi::CSCRPCDigi (int rpc, int pad, int bxn, int tbin){
  set(rpc, pad, bxn, tbin);
}

CSCRPCDigi::CSCRPCDigi (PackedDigiType packed_value){
  setData(packed_value);
}
/// Copy
CSCRPCDigi::CSCRPCDigi(const CSCRPCDigi& digi) {
  persistentData = digi.persistentData;
}
/// Default
CSCRPCDigi::CSCRPCDigi (){
  set(0,0,0,0);
}


/// Assignment
CSCRPCDigi& 
CSCRPCDigi::operator=(const CSCRPCDigi& digi){
  persistentData = digi.persistentData;
  return *this;
}

/// Getters

int CSCRPCDigi::getRpc()  const { return data()->rpc; }
int CSCRPCDigi::getPad()  const { return data()->pad; }
int CSCRPCDigi::getTbin() const { return data()->tbin; }
int CSCRPCDigi::getBXN()  const { return data()->bxn; }

 
/// Debug
void CSCRPCDigi::print() const {
  std::cout << "RPC = " << getRpc()
	    << "  Pad = " << getPad()
	    << "  Tbin = " << getTbin() 
	    << "  BXN = " << getBXN() << std::endl;
}

void CSCRPCDigi::dump() const {
  typedef std::bitset<8*sizeof(PackedDigiType)> bits;
  std::cout << *reinterpret_cast<const bits*>(data());  
}

/// Private members

void CSCRPCDigi::set(int rpc, int pad, int bxn, int tbin) {
  PackedDigiType* d = data();
  d->rpc   = rpc;
  d->pad   = pad;
  d->bxn   = bxn;
  d->tbin  = tbin;
}

CSCRPCDigi::PackedDigiType* CSCRPCDigi::data() {
  return reinterpret_cast<PackedDigiType*>(&persistentData);
}

const CSCRPCDigi::PackedDigiType* CSCRPCDigi::data() const {
  return reinterpret_cast<const PackedDigiType*>(&persistentData);
}

void CSCRPCDigi::setData(PackedDigiType p){
  *(data()) = p;
}
