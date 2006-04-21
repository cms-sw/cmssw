#ifndef ZTR_TRawScaler
#define ZTR_TRawScaler
#include "TObject.h"

class TRawScaler : public TObject {

protected:

  Int_t fN;          //which scaler
  Int_t fValues[12]; //scaler channel data

public:

  static Int_t fgNScalers;

  TRawScaler();
  TRawScaler(Int_t);
  TRawScaler(Int_t,Int_t*);
  virtual ~TRawScaler() {}
  Int_t    Compare(const TObject *obj) const { 
              if (fN < ((TRawScaler*)obj)->fN)
                return -1;
              else if (fN > ((TRawScaler*)obj)->fN)
                return 1;
              else
                return 0; }
  Int_t    GetN() const         { return fN;         }
  Int_t    GetValue(Short_t i) { return fValues[i]; }
  Int_t   *GetValues()          { return fValues;    }
  Bool_t   IsSortable()  const  { return kTRUE;      }
  virtual void     Print(const char *opt=0) const;
  void     SetValues(Int_t*);
  ClassDef(TRawScaler,1)  //cumulative counts of scintillators looking at the beam
};
#endif
