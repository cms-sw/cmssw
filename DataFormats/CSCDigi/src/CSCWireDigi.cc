/**\class CSCWireDigi
 *
 * Digi for CSC anode wires.
 * Based on modified DTDigi.
 *
 * $Date:$
 * $Revision:$
 *
 * \author N. Terentiev, CMU
 */

#include <DataFormats/CSCDigi/interface/CSCWireDigi.h>

#include <iostream>

using namespace std;

  /// Constructors

CSCWireDigi::CSCWireDigi (int wire, int tbin){
  wire_ = wire;
  tbin_ = tbin;
}
   /// Copy
CSCWireDigi::CSCWireDigi(const CSCWireDigi& digi) {
  wire_ = digi.getWireGroup();
  tbin_ = digi.getTimeBin();
}
   /// Default
CSCWireDigi::CSCWireDigi (){
  wire_ = 0;
  tbin_ = 0;
}


  /// Assignment
CSCWireDigi& 
CSCWireDigi::operator=(const CSCWireDigi& digi){
  wire_ = digi.getWireGroup();
  tbin_ = digi.getTimeBin();
  return *this;
}

  /// Debug

void CSCWireDigi::print() const {
  std::cout << " CSC Wire " << getWireGroup() 
       << " CSC Time Bin " << getTimeBin() << std::endl;
}
