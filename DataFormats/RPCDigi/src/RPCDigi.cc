/** \file
 * 
 *  $Date: 2006/04/06 07:54:05 $
 *  $Revision: 1.5 $
 *
 * \author Ilaria Segoni
 */


#include <DataFormats/RPCDigi/interface/RPCDigi.h>

#include <iostream>
#include <bitset>

using namespace std;


RPCDigi::RPCDigi(int strip, int bx){
	this->setStripBx( strip , bx );
}


/// Copy constructor
RPCDigi::RPCDigi(const RPCDigi& digi) {
  this->setStripBx(digi.strip(), digi.bx());
}



/// Assignment
RPCDigi& 
RPCDigi::operator=(const RPCDigi& digi){
	this->setStripBx(digi.strip(), digi.bx());
	return *this;
}

///Precedence operator
bool RPCDigi::operator<(const RPCDigi& digi) const{

if(digi.bx() == this->bx())
 return digi.strip()<this->strip();
else 
 return digi.bx()<this->bx();
 

}

/// Comparison
bool RPCDigi::operator == (const RPCDigi& digi) const {
  if ( !(this->strip() == digi.strip())     ||
       !(this->bx()== digi.bx()) ) return false;
  return true;
}

///Print Digi Content
void RPCDigi::print() const {
  cout << "strip " << this->strip() 
       << " bx   " << this->bx() << endl;
}

/// Getter methods:
int RPCDigi::strip() const { return strip_; }

int RPCDigi::bx() const { return bx_; }


/// Setter methods:

void RPCDigi::setStripBx(int strip, int bx) {
  strip_=strip;
  bx_=bx;
}

void RPCDigi::setStrip(int strip) {
  strip_=strip;
}

void RPCDigi::setBx(int bx) {
  bx_=bx;
}




