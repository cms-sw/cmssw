#ifndef ZTR_TRawTdcInfo
#define ZTR_TRawTdcInfo
#include "TObject.h"

class TRawTdcInfo : public TObject {

protected:

  Int_t  fNValue; //Number of values in fValues
  Int_t *fValues; //[fNValue] TDC measurements for clock-trig

  void Init();

public:

  TRawTdcInfo();
  TRawTdcInfo(Int_t);
  TRawTdcInfo(Int_t,Int_t*);
  virtual ~TRawTdcInfo();
  virtual void     Clear(const char *opt="");
  Int_t   *GetValues()             { return fValues;                }
  Int_t    GetNValue()             { return fNValue;                }
  Int_t    GetValue(Int_t n) const { n%=fNValue; return fValues[n]; }
  virtual void     Print(const char *opt=0) const;
  void     SetValues(Int_t,Int_t v[]);
  ClassDef(TRawTdcInfo,1) //TDC measurements for clock-trig
};
#endif
