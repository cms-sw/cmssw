/**\class CSCWireDigi
 *
 * Digi for CSC anode wires.
 * Based on modified DTDigi.
 *
 * $Date$
 * $Revision$
 *
 * \author N. Terentiev, CMU
 */

#include <DataFormats/CSCDigi/interface/CSCWireDigi.h>

#include <iostream>
#include <bitset>

using namespace std;

  /// Constructors

CSCWireDigi::CSCWireDigi (int wire, int tbin){
  set(wire, tbin);
}

CSCWireDigi::CSCWireDigi (ChannelType channel){
  ChannelPacking* ch = reinterpret_cast<ChannelPacking*>(&channel);
  set(ch->wire,
      ch->tbin);
}

CSCWireDigi::CSCWireDigi (PackedDigiType packed_value){
  setData(packed_value);
}
      /// Copy
CSCWireDigi::CSCWireDigi(const CSCWireDigi& digi) {
  persistentData = digi.persistentData;
}
      /// Default
CSCWireDigi::CSCWireDigi (){
  set(0,0);
}


  /// Assignment
CSCWireDigi& 
CSCWireDigi::operator=(const CSCWireDigi& digi){
  persistentData = digi.persistentData;
  return *this;
}

  /// Getters

CSCWireDigi::ChannelType
CSCWireDigi::channel() const {
  const PackedDigiType* d = data();
  ChannelPacking result;
  result.wire = d->wire;
  result.tbin = d->tbin;
  return *(reinterpret_cast<CSCWireDigi::ChannelType*>(&result));
}

int CSCWireDigi::getWireGroup() const { return data()->wire; }
int CSCWireDigi::getBeamCrossingTag() const { return data()->tbin; }

  /// Debug

void CSCWireDigi::print() const {
  cout << "Wire " << getWireGroup() 
       << "Tbin " << getBeamCrossingTag() <<endl;
}

void CSCWireDigi::dump() const {
  typedef bitset<8*sizeof(PackedDigiType)> bits;
  cout << *reinterpret_cast<const bits*>(data());  
}

  /// Private members

void CSCWireDigi::set(int  wire, int tbin) {
  PackedDigiType* d = data();
  d->wire   = wire;
  d->tbin   = tbin;
}

CSCWireDigi::PackedDigiType* 
CSCWireDigi::data() {
  return reinterpret_cast<PackedDigiType*>(&persistentData);
}

const CSCWireDigi::PackedDigiType* 
CSCWireDigi::data() const {
  return reinterpret_cast<const PackedDigiType*>(&persistentData);
}

void CSCWireDigi::setData(PackedDigiType p){
  *(data()) = p;
}
