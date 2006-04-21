//----------Author's Name:Jean Bourotte, Igor Semeniouk, Patrick Jarry (Windows porting)
//----------Copyright:Those valid for CMS sofware
//----------Modified:31/1/2003
#include <iostream>
#include "IORawData/Ecal2004TBInputService/interface/TRawCrystal.h"

using namespace std;

ClassImp(TRawCrystal)
//______________________________________________________________________________
//
//  Crystal raw data from sampling adc
//
//  fHeaders : To know version of ROSE modules used and other things
//                 for DAQ experts
//  fSamples : values of the sampling ADC
//
Int_t TRawCrystal::fgNSamplesCrystal = 10;

TRawCrystal::TRawCrystal() {
//Default constructor
  //printf( "TRawCrystal::TRawCrystal()\n" );
  Init();
}
TRawCrystal::TRawCrystal( Int_t h, Int_t s[] ){
//Constructor arguments:
//
// (1) - h    : for fHeader
// (2) - s    : values of the ns samplings
//
  //printf( "TRawCrystal::TRawCrystal( Int_t h, Int_t s[] )\n" );
  SetCrystalRaw( h, s );
}
TRawCrystal::~TRawCrystal() {
  Clear();
}
void TRawCrystal::Clear(const char *opt) {
  if (fSamples) delete [] fSamples;
  Init();
}
void TRawCrystal::Init() {
//Everything to 0
  fHeader  = 0;
  fNSample = 0;
  fSamples = 0;
}
void TRawCrystal::Print(const char *opt) const {
//Prints the whole class
  Short_t j;
  cout << endl;

  cout << "TCrystal fHeader       : ";
  cout << fHeader << "  ";
  cout << endl;
  cout << "Nb. of Samples : " << fNSample << endl;
  cout << "TCrystal Samples: ";
  for (j=0;j<fgNSamplesCrystal;j++) {
    cout << "  ";
    cout.width(12);
    cout << fSamples[j];
    if (!((j+1)%6)) {
      cout << endl;
      cout << "TCrystal Samples: ";
    }
  }
  cout << endl;
  cout << endl;
}
void TRawCrystal::SetCrystalRaw( Int_t h, Int_t s[] ) {
//Arguments:
//
// (1) - h    : for fHeaders
// (2) - s    : values of the ns samplings
//
  Short_t j;
  // Int_t ns = ( h & 0xff0 ) >> 4;
  Int_t ns = ( h & 0xff0 ) >> 8;
  if (ns>fgNSamplesCrystal) ns = fgNSamplesCrystal;
  fHeader = h;
  if ((ns<=0) && (fNSample != 0)) Clear();
  if ((ns >0) && (fNSample != ns)) {
    Clear();
    fNSample = ns;
    fSamples = new Int_t [fNSample];
  }
  for (j=0;j<ns;j++) fSamples[j] = s[j];
}

