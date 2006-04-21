#ifndef ZTR_TRawAdc2249
#define ZTR_TRawAdc2249
#include "TObject.h"

class TRawAdc2249 : public TObject {

protected:

  Int_t fN;          //which ADC
  Int_t fValue[12]; // q adc channel data

  void Init();

public:

  static Int_t fgNAdc2249s;  //actual number of ADC2249

  TRawAdc2249();
  TRawAdc2249(Int_t);
  TRawAdc2249(Int_t,Int_t*);
  virtual  ~TRawAdc2249() {}
  Int_t     Compare(const TObject *obj) const { 
              if (fN < ((TRawAdc2249*)obj)->fN)
                return -1;
              else if (fN > ((TRawAdc2249*)obj)->fN)
                return 1;
              else
                return 0; }
  Int_t     GetN() const         { return fN;        }
  Int_t    *GetValues()          { return fValue;    }
  Int_t     GetValue(Short_t j)  { return fValue[j]; }
  Bool_t    IsSortable() const   { return kTRUE;     }
  virtual void      Print(const char *opt=0) const;
  void      SetAdc(Int_t *);
  ClassDef(TRawAdc2249,1)  //2249 data
};
#endif
