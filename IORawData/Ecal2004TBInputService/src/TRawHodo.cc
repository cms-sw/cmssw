//----------Author's Name:Jean Bourotte, Igor Semeniouk, Patrick Jarry (Windows porting)
//----------Copyright:Those valid for CMS sofware
//----------Modified:31/3/2003
#include <iostream>
#include "IORawData/Ecal2004TBInputService/interface/TRawHodo.h"

using namespace std;

ClassImp(TRawHodo)
//______________________________________________________________________________
//
//  Hodoscope raw data
//
TRawHodo::TRawHodo() {
//Default constructor
  Init();
}

TRawHodo::TRawHodo(Int_t n ) {
//Constructor with number of elements.
  Short_t j;
  fNValue = n;
  fValues = new Int_t [fNValue];
  for (j=0;j<fNValue;j++) fValues[j] = 0;
}

TRawHodo::TRawHodo(Int_t n,Int_t d[]) {
//Constructor with data
  SetValues(n,d);
}

TRawHodo::~TRawHodo() {
  Clear();
}

void TRawHodo::Clear(const char *opt) {
  if (fValues) delete [] fValues;
  Init();
}

void TRawHodo::Init() {
//Everything to 0
  fNValue = 0;
  fValues = 0;
}

void TRawHodo::Print(const char *opt) const {
//Prints everything
  Int_t j;
  cout << endl;
  cout << "Nb. of data    : " << fNValue << endl;
  cout << "Data: ";
  for (j=0;j<fNValue;j++) {
    cout << "  " << fValues[j];
    if (!(j%8)) {
      cout << endl;
      cout << "Data: ";
    }
  }
  cout << endl;
  cout << endl;
}

void TRawHodo::SetValues(Int_t n,Int_t v[]) {
//Fill class variables
  Short_t j;
  if ((n<=0) && (fNValue != 0)) Clear();
  if ((n >0) && (fNValue != n)) {
    Clear();
    fNValue = n;
    fValues = new Int_t [fNValue];
  }
  for (j=0;j<n;j++) fValues[j] = v[j];
}


