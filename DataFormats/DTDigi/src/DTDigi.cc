/** \file
 * 
 *  $Date: 2010/03/03 15:53:20 $
 *  $Revision: 1.7 $
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


// DTDigi::DTDigi (ChannelType channel, int nTDC):
//   theWire(0),
//   theCounts(nTDC),
//   theNumber(0)
// {
//   theNumber = channel&number_mask;
//   theWire   = (channel&wire_mask)>>wire_offset;
// }


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
// DTDigi::ChannelType
// DTDigi::channel() const {
//   return  (theNumber & number_mask) | (theWire<<wire_offset)&wire_mask;
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

