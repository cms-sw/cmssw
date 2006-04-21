//----------Author's Name:Jean Bourotte, Igor Semeniouk, Patrick Jarry (Windows porting)
//----------Copyright:Those valid for CMS sofware
//----------Modified:31/1/2003
#include <iostream>
#include "IORawData/Ecal2004TBInputService/interface/TRawTriggerChannel.h"

using namespace std;

ClassImp(TRawTriggerChannel)
//______________________________________________________________________________
//
//  tdc measurements for trigger counters
//
TRawTriggerChannel::TRawTriggerChannel() {
//Default constructor.
  Init();
}
TRawTriggerChannel::TRawTriggerChannel(Int_t n) {
//Constructor with number of elements.
  Short_t j;
  fNValue = n;
  fValues = new Int_t [fNValue];
  for (j=0;j<fNValue;j++) fValues[j] = 0;
}

TRawTriggerChannel::TRawTriggerChannel(Int_t n,Int_t v[]) {
//Constructor with values
  Init();
  SetValues(n,v);
}

TRawTriggerChannel::~TRawTriggerChannel() {
  Clear();
}

void TRawTriggerChannel::Clear(const char *opt) {
  if (fValues) delete [] fValues;
  Init();
}

void TRawTriggerChannel::Init() {
//Everything to 0
  fNValue = 0;
  fValues = 0;
}

void TRawTriggerChannel::Print(const char *opt) const {
//Prints everything
  Short_t j;
  cout << endl;
  cout << "TRawTriggerChannel nv : " << fNValue << endl;
  cout << "Values:   ";
  for (j=0;j<fNValue;j++) cout << "  " << fValues[j];
  cout << endl;
  cout << endl;
}
void TRawTriggerChannel::SetValues(Int_t n,Int_t v[]) {
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
