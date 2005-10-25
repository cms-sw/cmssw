/** \file
 * 
 *  $Date: 2005/10/07 17:40:53 $
 *  $Revision: 1.1 $
 *
 * \author N. Amapane - INFN Torino
 */


#include <DataFormats/DTDigi/interface/DTDigi.h>

#include <iostream>
#include <bitset>

using namespace std;

const double DTDigi::reso =  25./32.; //ns

DTDigi::DTDigi (int wire, int number, int nTDC){
  set(wire, number, nTDC);
}


DTDigi::DTDigi (ChannelType channel, int nTDC){
  ChannelPacking* ch = reinterpret_cast<ChannelPacking*>(&channel);
  set(ch->wire,
      ch->number,
      nTDC);
}

DTDigi::DTDigi (PackedDigiType packed_value){
  setData(packed_value);
}

// Copy constructor
DTDigi::DTDigi(const DTDigi& digi) {
  persistentData = digi.persistentData;
}

DTDigi::DTDigi (){
  set(0,0,0);
}


// Assignment
DTDigi& 
DTDigi::operator=(const DTDigi& digi){
  persistentData = digi.persistentData;
  return *this;
}

// Comparison
bool
DTDigi::operator == (const DTDigi& digi) const {
  if ( !(wire() == digi.wire())     ||
       !(number()== digi.number()) ) return false;
  if ( countsTDC() != digi.countsTDC() ) return false;
  return true;
}

// Getters
DTDigi::ChannelType
DTDigi::channel() const {
  const PackedDigiType* d = data();
  ChannelPacking result;
  result.wire = d->wire;
  result.number= d->number;
  result.padding = 0;
  return *(reinterpret_cast<DTDigi::ChannelType*>(&result));
}

// DTEnum::ViewCode
// DTDigi::viewCode() const{
//   if ( slayer()==2 )
//     return DTEnum::RZed;
//   else return DTEnum::RPhi;
// }

double DTDigi::time() const { return countsTDC()*reso; }

int DTDigi::countsTDC() const { return data()->counts; }

int DTDigi::wire() const { return data()->wire; }

int DTDigi::number() const { return data()->number; }

// Setters

void DTDigi::setTime(double time){
  setCountsTDC(static_cast<int>(time/reso));
}

void DTDigi::setCountsTDC (int nTDC) {
  if (nTDC<0) cout << "WARNING: negative TDC count not supported "
		   << nTDC << endl;
  data()->counts = nTDC;
}

void DTDigi::setTrailer(int trailer) {
  data()->trailer=trailer;
}

// Debug

void
DTDigi::print() const {
  cout << "Wire " << wire() 
       << " Digi # " << number()
       << " Drift time (ns) " << time() << endl;
}

void
DTDigi::dump() const {
  typedef bitset<8*sizeof(PackedDigiType)> bits;
  cout << *reinterpret_cast<const bits*>(data());  
}

// ----- Private members

void
DTDigi::set(int  wire, int number, int counts) {

  PackedDigiType* d = data();
  d->wire   = wire;
  d->number = number;
  d->counts = counts;
  d->trailer = 0;
}

DTDigi::PackedDigiType* 
DTDigi::data() {
  return reinterpret_cast<PackedDigiType*>(&persistentData);
}

const DTDigi::PackedDigiType* 
DTDigi::data() const {
  return reinterpret_cast<const PackedDigiType*>(&persistentData);
}

void 
DTDigi::setData(PackedDigiType p){
  *(data()) = p;
}
