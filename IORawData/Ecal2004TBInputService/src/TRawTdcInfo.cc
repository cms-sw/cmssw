//----------Author's Name:Jean Bourotte, Igor Semeniouk, Patrick Jarry (Windows porting)
//----------Copyright:Those valid for CMS sofware
//----------Modified:31/3/2003
#include <iostream>
#include "IORawData/Ecal2004TBInputService/interface/TRawTdcInfo.h"
using namespace std;

ClassImp(TRawTdcInfo)
//______________________________________________________________________________
//
//  TDC measurements for clock-trig
//  In case of laser pulses, TRawTdcInfo gives the time of the maximum of the pulse. In
//the software, 2 types of histograms are provided: 
//  - The ones labelled "experimental" use the information in TRawTdcInfo to establish
//      the timing of the laser relative to the clock of the FADC. 
//  - The ones labelled "reconstructed" establish this timing by software, by mean
//      of fits on the pulses. 
//
TRawTdcInfo::TRawTdcInfo() {
//  Default constructor.
  Init();
}
TRawTdcInfo::TRawTdcInfo(Int_t n) {
//Constructor with number of elements
  Short_t j;
  fNValue = n;
  fValues = new Int_t [fNValue];
  printf( "TRawTdcInfo allocating %d int in Ctor(int)\n", fNValue );
  for (j=0;j<fNValue;j++) fValues[j] = 0;
}
TRawTdcInfo::TRawTdcInfo(Int_t n,Int_t v[]) {
//Constructor giving values
  Init();
  SetValues(n,v);
}
TRawTdcInfo::~TRawTdcInfo() {
//Destructor
  Clear();
}
void TRawTdcInfo::Clear(const char *opt) {
  if (fValues) {
    printf( "deleting %d int in Clear\n", fNValue );
    delete [] fValues;
  }
  Init();
}
void TRawTdcInfo::Init() {
//Everything to 0
  fNValue  = 0;
  fValues  = 0;
  //  printf( "end TRawTdcInfo::Init()\n" );
}
void TRawTdcInfo::Print(const char *opt) const {
  Short_t j;
  cout << endl;
  cout << "TRawTdcInfo nb val : " << fNValue << endl;
  cout << "Values:   ";
  for (j=0;j<fNValue;j++) cout << "  " << fValues[j];
  cout << endl;
  cout << endl;
}
void TRawTdcInfo::SetValues(Int_t n,Int_t v[]) {
//Set values to all variables of the class
  Short_t j;
  if ((n<=0) && (fNValue != 0)) Clear();
  if ((n >0) && (fNValue != n)) {
    Clear();
    fNValue = n;
    printf( "allocating %d int in SetValues(int)\n", fNValue );
    fValues = new Int_t [fNValue];
  }
  for (j=0;j<n;j++) fValues[j] = v[j];
}
