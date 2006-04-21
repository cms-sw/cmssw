//----------Author's Name:Jean Bourotte, Igor Semeniouk, Patrick Jarry (Windows porting)
//----------Copyright:Those valid for CMS sofware
//----------Modified:8/3/2003
#include <iostream>
#include "IORawData/Ecal2004TBInputService/interface/TRawPn.h"

using namespace std;

ClassImp(TRawPn)
//______________________________________________________________________________
//
// This class has to be used instead of the class TRawAdc2249 when the PN diodes of
//the monitoring are read by sampling ADCs instead of normal 2249 ADCs.
//
Int_t TRawPn::fgNPns = 10;

TRawPn::TRawPn() {
//Default constructor
  Init();
}
TRawPn::TRawPn(Int_t n) {
//Constructor with Pn numb
  Init();
  fN = n;
}
TRawPn::TRawPn(Int_t n,Int_t ns,Int_t s[])
// Constructor with setting of everything
{
  fN = n;
  SetPn(ns,s);
  fVInj = 0;
}
TRawPn::~TRawPn() {
  if (fSamples) delete [] fSamples;
}
void TRawPn::Remove() {
  if (fSamples) delete [] fSamples;
  Init();
}
Int_t* TRawPn::GetSamples(Int_t &n) {
//Give access to coefficients of polynom and their number
  n = fNSample;
  return fSamples;
}
void TRawPn::Init() {
//Initialization
  fN         = -1;
  fNSample = 0;
  fSamples = 0;
}
void TRawPn::Print(const char *opt) const {
//Prints everything
  Int_t j;
  cout << endl;
  cout << "Pn number     : " << fN << endl;
  cout << "Nb. of Samples: " << fNSample << endl;
  cout << "Samples: ";
  for (j=0;j<fNSample;j++) {
    cout << "  " << fSamples[j];
    if (!(j%8)) {
      cout << endl;
      cout << "Samples: ";
    }
  }
  cout << endl;
  cout << endl;
}
void TRawPn::SetPn(Int_t ns,Int_t s[]) {
//Defines the whole class
  Short_t j;
  if ((ns<=0) && (fNSample != 0)) Remove();
  if ((ns >0) && (fNSample != ns)) {
    Remove();
    fNSample = ns;
    fSamples = new Int_t [fNSample];
  }
  for (j=0;j<ns;j++) fSamples[j] = s[j];
}


