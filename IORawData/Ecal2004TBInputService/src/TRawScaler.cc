//----------Author's Name:Jean Bourotte, Igor Semeniouk, Patrick Jarry (Windows porting)
//----------Copyright:Those valid for CMS sofware
//----------Modified:31/3/2003
#include <iostream>
#include "IORawData/Ecal2004TBInputService/interface/TRawScaler.h"
using namespace std;

ClassImp(TRawScaler)

Int_t TRawScaler::fgNScalers  = 3;
//______________________________________________________________________________
//
// TRawScalers of type 2551
// This class registers essentially cumulative counts of scintillators looking at
//the beam.
//
//  1st scaler :
//
//  fValues[2]    : S2 : 2X2 mm counter in the beam, included in the trigger 
//  fValues[6]    : S6 : 20X20 mm counter in the beam 
//  fValues[7]    : halo counter 
//  fValues[9]    : muon counter
//
//  2nd scaler :
//
//  fValues[1-12] : finger counters to center the beam horizontally
//
//  3rd scaler :
//
//  fValues[1-12] : finger counters to center the beam vertically
//
TRawScaler::TRawScaler() {
//Default constructor
  for(Short_t j=0;j<12;j++) fValues[j] = 0;
}
TRawScaler::TRawScaler(Int_t n) {
//Constructor with scaler number
  fN = n;
  for(Short_t j=0;j<12;j++) fValues[j] = 0;
}
TRawScaler::TRawScaler(Int_t n,Int_t c[]) {
  fN = n;
  for(Short_t j=0;j<12;j++) fValues[j] = c[j];
}
void TRawScaler::Print(const char *opt) const {
//Prints everything
  Short_t j;
  cout << endl;
  cout << "TRawScaler number : " << fN << endl;
  cout << endl;
  cout << "TRawScaler: ";
  for (j=0;j<12;j++) cout << "  " << fValues[j];
  cout << endl;
  cout << endl;
}
void TRawScaler::SetValues(Int_t c[]) {
  for(Short_t j=0;j<12;j++) fValues[j] = c[j];
}
