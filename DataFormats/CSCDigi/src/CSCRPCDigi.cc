/**\class CSCRPCDigi
 *
 * Digi for RPC data coming thru RAT-ALCT-DDU.
 *
 * $Date$
 * $Revision$
 *
 * \author N. Terentiev, CMU
 */

#include <DataFormats/CSCDigi/interface/CSCRPCDigi.h>

#include <iostream>
#include <bitset>

using namespace std;

  /// Constructors

CSCRPCDigi::CSCRPCDigi (int strip, int tbin){
  set(strip, tbin);
}

CSCRPCDigi::CSCRPCDigi (ChannelType channel){
  ChannelPacking* ch = reinterpret_cast<ChannelPacking*>(&channel);
  set(ch->strip,
      ch->tbin);
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
  set(0,0);
}


  /// Assignment
CSCRPCDigi& 
CSCRPCDigi::operator=(const CSCRPCDigi& digi){
  persistentData = digi.persistentData;
  return *this;
}

  /// Getters

CSCRPCDigi::ChannelType
CSCRPCDigi::channel() const {
  const PackedDigiType* d = data();
  ChannelPacking result;
  result.strip = d->strip;
  result.tbin  = d->tbin;
  return *(reinterpret_cast<CSCRPCDigi::ChannelType*>(&result));
}

int CSCRPCDigi::getStrip() const { return data()->strip; }
int CSCRPCDigi::getBx() const { return data()->tbin; }

  /// Debug

void CSCRPCDigi::print() const {
  cout << "RPC strip" << getStrip() 
       << "Tbin " << getBx() <<endl;
}

void CSCRPCDigi::dump() const {
  typedef bitset<8*sizeof(PackedDigiType)> bits;
  cout << *reinterpret_cast<const bits*>(data());  
}

  /// Private members

void CSCRPCDigi::set(int strip, int tbin) {
  PackedDigiType* d = data();
  d->strip   = strip;
  d->tbin   = tbin;
}

CSCRPCDigi::PackedDigiType* 
CSCRPCDigi::data() {
  return reinterpret_cast<PackedDigiType*>(&persistentData);
}

const CSCRPCDigi::PackedDigiType* 
CSCRPCDigi::data() const {
  return reinterpret_cast<const PackedDigiType*>(&persistentData);
}

void CSCRPCDigi::setData(PackedDigiType p){
  *(data()) = p;
}
