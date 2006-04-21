#ifndef ZTR_TRawHodo
#define ZTR_TRawHodo
#include "TObject.h"

class TRawHodo : public TObject {

protected:

  Int_t  fNValue;
  Int_t  *fValues; //[fNValue] raw data fibers hodoscopes

  void Init();

public:

  TRawHodo();
  TRawHodo(Int_t );
  TRawHodo(Int_t,Int_t*);
  virtual ~TRawHodo();
  virtual void     Clear(const char *opt="");
  Int_t   *GetValues() { return fValues; }
  Int_t    GetValue( Int_t j ) { return fValues[j]; }
  Int_t    GetLen() { return fNValue; }
  virtual void     Print(const char *opt=0) const;
  void     SetValues(Int_t,Int_t*);
  ClassDef(TRawHodo,1)  //Hodoscope raw data
};
#endif
