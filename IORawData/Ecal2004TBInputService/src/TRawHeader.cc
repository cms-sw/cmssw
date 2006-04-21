//----------Author's Name:Jean Bourotte, Igor Semeniouk, Patrick Jarry (Windows porting)
//----------Copyright:Those valid for CMS sofware
//----------Modified:31/1/2003
#include <iostream>
#include "IORawData/Ecal2004TBInputService/interface/TRawHeader.h"

using namespace std;
ClassImp(TRawHeader)
//______________________________________________________________________________
//
// TRawHeader.  Look at the class variables to know the content
//
TRawHeader::TRawHeader() {
//Default constructor
  Init();
}

void TRawHeader::Init() {
//Everything to 0
  //fRunNum              = 0;
  fBurstNum            = 0;
  fEvtNum              = 0;
  //fRunType             = 0;
  fDate                = 0;
  fTrigMask            = 0;
  //fNMod                = 0;
  //fNChMod              = 0;
  //fFrameLength         = 0;
  fFPPAMode            = 0;
  fPNMode              = 0;
  //fROSEMode            = 0;
  fThetaTableIndex = 0;
  fPhiTableIndex = 0;
  fLightIntensityIndex = 0;
  fInBeamXtal          = 0;
}
void TRawHeader::Print(const char *opt) const {
  cout << endl;
//Print this header
  cout << "fBurstNum     = ";
  cout.width(12);
  cout << fBurstNum;
  cout << "    fEvtNum       = ";
  cout.width(12);
  cout << fEvtNum << endl;
//
  cout << "fDate         = ";
  cout.width(12);
  cout << fDate;
  cout << "    fTrigMask     = ";
  cout.width(12);
  cout << fTrigMask << endl;
//
  cout << "fLightII      = ";
  cout.width(12);
  cout << fLightIntensityIndex;
  cout << "    fInBeamXtal   = ";
  cout.width(12);
  cout << fInBeamXtal << endl;
}

void TRawHeader::Set(Int_t iv[] ) { 
  fBurstNum            = iv[0];
  fEvtNum              = iv[1];
  fDate                = iv[2];
  fTrigMask            = iv[3];
  fFPPAMode            = iv[4];
  fLightIntensityIndex = iv[5];
  fInBeamXtal          = iv[6];
  fThetaTableIndex     = iv[7];
  fPhiTableIndex       = iv[8];
  fPNMode              = iv[9];
}
