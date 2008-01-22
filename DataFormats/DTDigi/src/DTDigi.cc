/** \file
 * 
 *  $Date: 2006/04/05 15:03:07 $
 *  $Revision: 1.4 $
 *
 * \author N. Amapane - INFN Torino
 */


#include <DataFormats/DTDigi/interface/DTDigi.h>


using namespace std;


const double DTDigi::reso =  25./32.; //ns


DTDigi::DTDigi (int wire, int nTDC, int number) : 
  theWire(wire),
  theCounts(nTDC),
  theNumber(number)
{}


DTDigi::DTDigi (int wire, double tdrift, int number): 
  theWire(wire),
  theCounts(static_cast<int>(tdrift/reso)),
  theNumber(number)
{}


DTDigi::DTDigi (ChannelType channel, int nTDC):
  theWire(0),
  theCounts(nTDC),
  theNumber(0)
{
  ChannelPacking* ch = reinterpret_cast<ChannelPacking*>(&channel);
  theWire = ch->wire;
  theNumber = ch->number;
}


DTDigi::DTDigi ():
  theWire(0),
  theCounts(0), 
  theNumber(0)
{}


// Comparison
bool
DTDigi::operator == (const DTDigi& digi) const {
  if ( theWire != digi.wire() ||
       //       theNumber != digi.number() || //FIXME required ??
       theCounts != digi.countsTDC() ) return false;
  return true;
}

// Getters
DTDigi::ChannelType
DTDigi::channel() const {
  ChannelPacking result;
  result.wire = theWire;
  result.number= theNumber;
  return *(reinterpret_cast<DTDigi::ChannelType*>(&result));
}

// DTEnum::ViewCode
// DTDigi::viewCode() const{
//   if ( slayer()==2 )
//     return DTEnum::RZed;
//   else return DTEnum::RPhi;
// }

double DTDigi::time() const { return theCounts*reso; }

uint32_t DTDigi::countsTDC() const { return theCounts; }

int DTDigi::wire() const { return theWire; }

int DTDigi::number() const { return theNumber; }

// Setters

void DTDigi::setTime(double time){
  theCounts = static_cast<int>(time/reso);
}

void DTDigi::setCountsTDC (int nTDC) {
  if (nTDC<0) cout << "WARNING: DTDigi::setCountsTDC: negative TDC count not supported "
		   << nTDC << endl;
  theCounts = nTDC;
}


// Debug

void
DTDigi::print() const {
  cout << "Wire " << wire() 
       << " Digi # " << number()
       << " Drift time (ns) " << time() << endl;
}

