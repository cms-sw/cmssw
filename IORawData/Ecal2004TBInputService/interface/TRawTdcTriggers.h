#ifndef ZTR_TRawTdcTriggers
#define ZTR_TRawTdcTriggers
#include "TObject.h"

class TRawTdcTriggers : public TObject {

protected:

  Int_t    fNValue; //Number of values in fValues
  Int_t   *fValues; //[fNValue] TDC measurements for trigger counters

  void Init();

public:

  TRawTdcTriggers();
  TRawTdcTriggers(Int_t);
  TRawTdcTriggers(Int_t,Int_t*);
  virtual ~TRawTdcTriggers();
  virtual void     Clear(const char *opt="");
  Int_t   *GetValues()              { return fValues; }
  Int_t    GetValue( Int_t k )               { return fValues[k]; }
  Int_t    GetLen() { return fNValue; }
  virtual void     Print(const char *opt=0) const;
  void     SetValues( Int_t, Int_t* );
  ClassDef( TRawTdcTriggers, 1 ) //tdc measurements for trigger counters
};
#endif
