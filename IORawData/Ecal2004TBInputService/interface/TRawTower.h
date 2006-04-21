#ifndef ZTR_TRawTower
#define ZTR_TRawTower
#include "TClonesArray.h"
#include "TRawCrystal.h"
#include "TRawTriggerChannel.h"

class TRawTower : public TObject {

protected:

  Int_t         fN;              //Tower number
  Short_t       fNRose;          //Rose number for that tower
  Short_t       fNChRose;        //Rose channel number for that tower
  Int_t         fTowerHeader[4]; //Tower Header words (3 needed)
  Int_t         fStripHeader[5]; //Strip Header words

  void          Init();

public:

  Int_t         fNCrystal;      //Number of TRawCrystal objects in fCrystalsRaw
  // TRawCrystal   fCrystalsRaw[N_CRYSINTOWER]; //->Collection of TRawCrystal objects
  TRawCrystal   fCrystalsRaw[25]; //->Collection of TRawCrystal objects
  TRawTriggerChannel   fRawTriggerChannel;

  static Int_t  fgNTowers;      //Actual number of towers

  //  static TClonesArray *fgCrystalsRaw[N_TOWERS]; //static version of fCrystalsRaw for all TRawTowers;

  TRawTower();
  TRawTower(Int_t,Short_t,Short_t);
  virtual ~TRawTower() {}
  //TRawCrystal *AddCrystal(Int_t h, Int_t s[]);
  virtual void         Clear(const char *opt="");
  Int_t        GetN()                    { return fN;              }
  Int_t        GetTowerHeader(Short_t j) { return fTowerHeader[j]; }
  Int_t        GetStripHeader(Short_t j) { return fStripHeader[j]; }
  TRawCrystal  *GetCrystal(Short_t n)    { return &fCrystalsRaw[n]; }
  void         SetHeaders( Int_t*, Int_t* );
  void         SetRoseInfo( Short_t, Short_t );
  Short_t      GetNRose()                { return fNRose;          }
  Short_t      GetNChRose()              { return fNChRose;        }
  virtual void         Print(const char *opt="") const;
  void         SetN(Int_t n)             { fN = n;                 }
  //void         SetCrystalsRaw(Short_t j) { fCrystalsRaw = fgCrystalsRaw[j]; }
  void         FillCrystalData( Int_t, Int_t, Int_t* );
  ClassDef(TRawTower,2)  //Tower of 25 crystals
};
#endif
