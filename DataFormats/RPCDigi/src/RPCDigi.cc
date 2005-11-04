/** \file
 * 
 *  $Date: 2005/11/03 15:25:32 $
 *  $Revision: 1.1 $
 *
 * \author Ilaria Segoni
 */


#include <DataFormats/RPCDigi/interface/RPCDigi.h>

#include <iostream>
#include <bitset>

using namespace std;


RPCDigi::RPCDigi(int strip, int bx){
    setStripBx( strip , bx );
}


/// Copy constructor
//RPCDigi::RPCDigi(const RPCDigi& digi) {
//  persistentData = digi.persistentData;
//}



/// Assignment
//RPCDigi& 
//RPCDigi::operator=(const RPCDigi& digi){
 // persistentData = digi.persistentData;
 // return *this;
//}

/// Comparison
bool RPCDigi::operator == (const RPCDigi& digi) const {
  if ( !(strip() == digi.strip())     ||
       !(bx()== digi.bx()) ) return false;
  return true;
}

///Print Digi Content
void RPCDigi::print() const {
  cout << "strip " << strip() 
       << " bx   " << bx() << endl;
}

/// Getter methods:
int RPCDigi::strip() const { return data()->strip; }

int RPCDigi::bx() const { return data()->bx; }


/// Setter methods:

void RPCDigi::setStripBx(int strip, int bx) {
  data()->strip=strip;
  data()->bx=bx;
}

void RPCDigi::setStrip(int strip) {
  data()->strip=strip;
}

void RPCDigi::setBx(int bx) {
  data()->bx=bx;
}




///  Private members

RPCDigi::PackedDigiType* 
RPCDigi::data() {
  return reinterpret_cast<PackedDigiType*>(&persistentData);
}

const RPCDigi::PackedDigiType* 
RPCDigi::data() const {
  return reinterpret_cast<const PackedDigiType*>(&persistentData);
}
