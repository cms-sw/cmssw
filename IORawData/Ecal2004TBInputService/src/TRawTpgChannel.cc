//----------Author's Name:Jean Bourotte, Igor Semeniouk, Patrick Jarry (Windows porting)
//----------Copyright:Those valid for CMS sofware
//----------Modified:31/1/2003
//----------Modified; 8/11/2004 Patrick Jarry (tRawTpgChannel)

#include <iostream>
#include "IORawData/Ecal2004TBInputService/interface/TRawTpgChannel.h"

using namespace std;

ClassImp(TRawTpgChannel)
//______________________________________________________________________________
//
//  tdc measurements for Tpg counters
//
TRawTpgChannel::TRawTpgChannel() {
//Default constructor.
  Init();
}
TRawTpgChannel::TRawTpgChannel(Int_t n) {
//Constructor with number of elements.
  Short_t j;
  fNValue = n;
  fValues = new Int_t [fNValue];
  for (j=0;j<fNValue;j++) fValues[j] = 0;
}

TRawTpgChannel::TRawTpgChannel(Int_t n,Int_t v[]) {
//Constructor with values
  Init();
  SetValues(n,v);
}

TRawTpgChannel::~TRawTpgChannel() {
  Clear();
}

void TRawTpgChannel::Clear() {
  if (fValues) delete [] fValues;
  Init();
}

void TRawTpgChannel::Init() {
//Everything to 0
  fNValue = 0;
  fValues = 0;
}

void TRawTpgChannel::Print() const {
//Prints everything
  Short_t j;
  cout << endl;
  cout << "TRawTpgChannel nv : " << fNValue << endl;
  cout << "Values:   ";
  for (j=0;j<fNValue;j++) cout << "  " << fValues[j];
  cout << endl;
  cout << endl;
}
void TRawTpgChannel::SetValues(Int_t n,Int_t v[]) {
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
void TRawTpgChannel::SetTpgChannel(Int_t ns,Int_t s[]) {
//Defines the whole class
  Short_t j;
  if ((ns<=0) && (fNValue != 0)) Clear();
  if ((ns >0) && (fNValue != ns)) {
    Clear();
    fNValue = ns;
    fValues = new Int_t [fNValue];
  }
  for (j=0;j<ns;j++) fValues[j] = s[j];
}


