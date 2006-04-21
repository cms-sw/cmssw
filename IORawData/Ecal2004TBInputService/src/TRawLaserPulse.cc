//----------Author's Name:Jean Bourotte, Igor Semeniouk, Patrick Jarry (Windows porting)
//----------Copyright:Those valid for CMS sofware
//----------Modified:8/3/2003
#include <iostream>
#include "IORawData/Ecal2004TBInputService/interface/TRawLaserPulse.h"
using namespace std;

ClassImp(TRawLaserPulse)
//______________________________________________________________________________
//
// This class has to be used instead of the class TRawAdc2249 when the PN diodes of
//the monitoring are read by sampling ADCs instead of normal 2249 ADCs.
//
TRawLaserPulse::TRawLaserPulse() {
//Default constructor
  Init();
}

TRawLaserPulse::TRawLaserPulse(Int_t ns,Int_t s[])
// Constructor with setting of everything
{
  SetLaserPulse(ns,s);
}

TRawLaserPulse::~TRawLaserPulse() {
  if (fSamples) delete [] fSamples;
}

void TRawLaserPulse::Remove() {
  if (fSamples) delete [] fSamples;
  Init();
}

Int_t* TRawLaserPulse::GetSamples() {
  return fSamples;
}

void TRawLaserPulse::Init() {
//Initialization
  fNSample = 0;
  fSamples = 0;
}

void TRawLaserPulse::Print() const {
//Prints everything
  Int_t j;
  cout << endl;
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

void TRawLaserPulse::SetLaserPulse(Int_t ns,Int_t s[]) {
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


