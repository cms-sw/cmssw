#ifndef ZTR_TRawTpgChannel
#define ZTR_TRawTpgChannel
#include "TObject.h"

class TRawTpgChannel : public TObject {

protected:

  Int_t    fNValue; //Number of values in fValues
  Int_t   *fValues; //[fNValue] TDC measurements for tpg counters

  void Init();

public:

  TRawTpgChannel();
  TRawTpgChannel(Int_t);
  TRawTpgChannel(Int_t,Int_t*);
  virtual ~TRawTpgChannel();
  void     Clear();
  Int_t   *GetValues()              { return fValues; }
  Int_t    GetValue( Int_t k )               { return fValues[k]; }
  //Int_t    GetValue(Int_t n) const { n%=fNValue; return fValues[n]; }
  Int_t    GetLen() { return fNValue; }
  void     Print() const;
  void     SetValues( Int_t, Int_t* );
  void     SetTpgChannel(Int_t ,Int_t*) ;
  ClassDef( TRawTpgChannel, 1 ) // tpg counters
};
#endif
