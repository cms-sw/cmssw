//----------Author's Name:Jean Bourotte, Igor Semeniouk, Patrick Jarry (Windows porting)
//----------Copyright:Those valid for CMS sofware
//----------Modified:31/3/2003
#include <iostream>
#include "IORawData/Ecal2004TBInputService/interface/TRawPattern.h"
using namespace std;

ClassImp(TRawPattern)
//______________________________________________________________________________
//
//  Content of the 3 pattern units used in the test beam 
//
TRawPattern::TRawPattern() {
//Default constructor
  Init();
}

TRawPattern::TRawPattern(Int_t n, Int_t v[]) {
//Constructor with values
  Init();
  SetValues(n,v);
}

TRawPattern::~TRawPattern() {
  Clear();
}

void TRawPattern::Init() {
//Everything to 0
  fNValue = 0;
  fValues = 0;
}

void TRawPattern::Clear(const char *opt) {
  if (fValues) delete [] fValues;
  Init();
}


void TRawPattern::Print(const char *opt) const {
  Short_t j;
  cout << endl;

  for (j=0;j<fNValue;j++) cout << "  " << fValues[j];
  cout << endl;
  cout << endl;
}

void TRawPattern::SetValues( Int_t n, Int_t v[]) {
  Short_t j;
  if ((n<=0) && (fNValue != 0)) Clear();
  if ((n >0) && (fNValue != n)) {
    Clear();
    fNValue = n;
    fValues = new Int_t [fNValue];
  }
  for (j=0;j<n;j++) fValues[j] = v[j];
}
