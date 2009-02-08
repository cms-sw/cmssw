//----------Author's Name:Jean Bourotte, Igor Semeniouk, Patrick Jarry (Windows porting)
//----------Copyright:Those valid for CMS sofware
//----------Modified:31/1/2003
//#include "config.h"
#include <iostream>
#include "IORawData/Ecal2004TBInputService/interface/TRunInfo.h"

using namespace std;

ClassImp(TRunInfo)
//______________________________________________________________________________
//
// TRunInfo.  Look at the class variables to know the content
//
// const char *gRawRootVersion=VERSION;

TRunInfo::TRunInfo(){
  Init();
}

void TRunInfo::Init() {
//Everything to 0
  fRunNum        = 0;
  fRunType       = 0;
  fNTowersMax    = 0;

  fNMod          = 0;
  fNChMod        = 0;
  fROSEMode      = 0;
  fFrameLength   = 0;

  fNPNs          = 0;
  fFrameLengthPN = 0;

  fSoftVersion   = 0;
}

void TRunInfo::Print(const char *opt) const {
  cout << endl;
//Print RunInfo
  cout << "fRunNum        = ";
  cout.width(12);
  cout << fRunNum  << "    ";
  cout << "fRunType       = ";
  cout.width(12);
  cout << fRunType << endl;

  cout << "fNTowersMax    = ";
  cout.width(12);
  cout << fNTowersMax << "    ";
  cout << "fNMod          = ";
  cout.width(12);
  cout << fNMod    << endl;

  cout << "fNChMod        = ";
  cout.width(12);
  cout << fNChMod << "    ";
  cout << "fROSEMode      = ";
  cout.width(12);
  cout << fROSEMode << endl;

  cout << "fFrameLength   = ";
  cout.width(12);
  cout << fFrameLength << "    ";
  cout << "fNPNs          = ";
  cout.width(12);
  cout << fNPNs << endl;

  cout << "fFrameLengthPN = ";
  cout.width(12);
  cout << fFrameLengthPN << endl;
}

void TRunInfo::Set(Int_t iv[] ) {
  fRunNum        = iv[0];
  fRunType       = iv[1];
  fNTowersMax    = iv[2];

  fNMod          = iv[3];
  fNChMod        = iv[4];
  fROSEMode      = iv[5];
  fFrameLength   = iv[6];

  fNPNs          = iv[7];
  fFrameLengthPN = iv[8];

  fSoftVersion   = iv[9];
}

void TRunInfo::SetNTowMax(Int_t n ) 
{
  fNTowersMax    = n;
}
