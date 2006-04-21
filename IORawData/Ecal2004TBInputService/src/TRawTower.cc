//----------Author's Name:Jean Bourotte, F.X.Gentit
//----------Copyright:Those valid for CMS sofware
//----------Modified:10/4/2003
#include <iostream>
#include "IORawData/Ecal2004TBInputService/interface/TRawTower.h"

using namespace std;

ClassImp(TRawTower)

//TClonesArray *TRawTower::fgCrystalsRaw[N_TOWERS];
//______________________________________________________________________________
//
//  Tower of 25 crystals
//
Int_t TRawTower::fgNTowers = 68;

TRawTower::TRawTower() {
//Default constructor
  Init();
  //  printf("TRawTower constructor\n");
}

TRawTower::TRawTower(Int_t n,  Short_t rose_n,  Short_t rose_ch ) {
//Constructor with data
  fNCrystal = 0;
  fN = n;
  fNRose   = rose_n;
  fNChRose = rose_ch; 
}

void TRawTower::FillCrystalData( Int_t icrys, Int_t h, Int_t *s ) {
  if (icrys < 0 ||  icrys > 24) {
    printf( "TRawTowers---> icrys is bad !!!!!!!!!!!!!! %d\n", icrys );
  } else {
    fCrystalsRaw[icrys].SetCrystalRaw( h, s );
    fNCrystal++;
  }
}

void TRawTower::Clear(const char *opt) {
//Clear to go to next event
  fNCrystal = 0;
  //fCrystalsRaw->Clear("C");
}

void TRawTower::Init() {
//Initialization
  Short_t j;
  fN = -1;
  fNCrystal = 0;
  fNRose    = 0;
  fNChRose  = 0;
  for (j=0;j<4;j++) fTowerHeader[j] = 0;
  for (j=0;j<5;j++) fStripHeader[j] = 0;
}

void TRawTower::SetHeaders( Int_t *th, Int_t *sh ){
  Short_t j;
  for (j=0;j<4;j++) fTowerHeader[j] = th[j];
  for (j=0;j<5;j++) fStripHeader[j] = sh[j];
}

void TRawTower::SetRoseInfo( Short_t rose_n,  Short_t rose_ch ){
  fNRose   = rose_n;
  fNChRose = rose_ch; 
}

void TRawTower::Print(const char *opt) const {
//Prints everything
  Int_t j;
  cout << endl;
  cout << "Tower Number   : " << fN << endl;
  cout << "Nb of crystals : " << fNCrystal << endl;
  cout << "Rose Number    : " << fNRose << endl;
  cout << "Rose Channel   : " << fNChRose << endl;
  cout << "Tower Header words:" << endl;
  for (j=0;j<4;j++) {
    cout << "   " << fTowerHeader[j];
  }
  cout << endl;
  cout << "Strip Header words:" << endl;
  for (j=0;j<5;j++) {
    cout << "   " << fStripHeader[j];
  }
  cout << endl;
  //for (j=0;j<fNCrystal;j++) (*fCrystalsRaw)[j]->Print();
  //call only TObect print method
  for (j=0;j<fNCrystal;j++){
    fCrystalsRaw[j].Print();
  }
  cout << endl;
}
