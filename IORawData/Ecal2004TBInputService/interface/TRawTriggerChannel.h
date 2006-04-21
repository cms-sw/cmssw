#ifndef ZTR_TRawTriggerChannel
#define ZTR_TRawTriggerChannel
#include "TObject.h"

class TRawTriggerChannel : public TObject {

protected:

  Int_t    fNValue; //Number of values in fValues
  Int_t   *fValues; //[fNValue] TDC measurements for trigger counters

  void Init();

public:

  TRawTriggerChannel();
  TRawTriggerChannel(Int_t);
  TRawTriggerChannel(Int_t,Int_t*);
  virtual ~TRawTriggerChannel();
  virtual void     Clear(const char *opt="");
  Int_t   *GetValues()              { return fValues; }
  Int_t    GetValue( Int_t k )               { return fValues[k]; }
  Int_t    GetLen() { return fNValue; }
  void     Print(const char *opt=0) const;
  void     SetValues( Int_t, Int_t* );
  ClassDef( TRawTriggerChannel, 1 ) //tdc measurements for trigger counters
};
#endif
