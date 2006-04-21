//----------Author's Name:Jean Bourotte, Igor Semeniouk, Patrick Jarry (Windows porting)
//----------Copyright:Those valid for CMS sofware
//----------Modified:31/3/2003
#include <iostream>
#include "IORawData/Ecal2004TBInputService/interface/TRawAdc2249.h"

using namespace std;
ClassImp(TRawAdc2249)
//______________________________________________________________________________
//
Int_t TRawAdc2249::fgNAdc2249s = 2;
TRawAdc2249::TRawAdc2249() {
//Default constructor
  Init();
}
TRawAdc2249::TRawAdc2249(Int_t n) {
//Constructor giving Adc number
  Init();
  fN = n;
}
TRawAdc2249::TRawAdc2249(Int_t n,Int_t h[]) {
  Short_t j;
  fN = n;
  for(j=0;j<12;j++) fValue[j]=h[j];
}
void TRawAdc2249::Init() {
  Short_t j;
  for (j=0;j<12;j++) fValue[j] = 0;
}
void TRawAdc2249::Print(const char *opt) const {
//Prints everything
  Short_t j;
  cout << endl;
  cout << "TAdc224 number  : " << fN << endl;
  cout << "Q ADC: ";
  for (j=0;j<12;j++) cout << "  " << fValue[j];
  cout << endl;
  cout << endl;
}
void TRawAdc2249::SetAdc(Int_t h[]) {
  Short_t j;
  for(j=0;j<12;j++) fValue[j]=h[j];
}
