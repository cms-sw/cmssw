#ifndef ZTR_TRawPn
#define ZTR_TRawPn
#include "TObject.h"

class TRawPn : public TObject {

protected:

  Int_t   fN;                    //Number of pn blocks.
  Int_t   fNSample;              //Number of samples
  Int_t  *fSamples;              //[fNSample] sadc samples
  Int_t   fVInj;                 //in case of pulse injection Vinj*1000

  void Init();

public:

  static Int_t fgNPns;

  TRawPn();
  TRawPn(Int_t);
  TRawPn(Int_t,Int_t,Int_t*);
  virtual ~TRawPn();
  void     Remove();
  Int_t    Compare(const TObject *obj) const { 
              if (fN < ((TRawPn*)obj)->fN)
                return -1;
              else if (fN > ((TRawPn*)obj)->fN)
                return 1;
              else
                return 0; }
  Int_t    GetN() const               { return fN;          }
  Int_t    GetNSamples()              { return fNSample; }
  Int_t    GetSample(Int_t k) const   { return fSamples[k]; }
  Int_t   *GetSamples(Int_t&);
  Int_t    GetVInj()                  { return fVInj; }
  void     SetVInj( Int_t v )         { fVInj = v; }
  Bool_t   IsSortable()  const        { return kTRUE;       }
  void     Print(const char *opt=0) const;
  void     SetPn(Int_t,Int_t*);
  ClassDef(TRawPn,2) //A pn diode data
};
#endif
