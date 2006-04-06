/**\class CSCWireDigi
 *
 * Digi for CSC anode wires.
 * Based on modified DTDigi.
 *
 * $Date: 2006/04/05 19:40:16 $
 * $Revision: 1.2 $
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

   /// Default
CSCWireDigi::CSCWireDigi (){
  wire_ = 0;
  tbin_ = 0;
}



  /// Debug

void CSCWireDigi::print() const {
  std::cout << " CSC Wire " << getWireGroup() 
       << " CSC Time Bin " << getTimeBin() << std::endl;
}
