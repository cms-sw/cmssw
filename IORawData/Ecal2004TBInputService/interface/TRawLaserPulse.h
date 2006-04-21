#ifndef ZTR_TRawLaserPulse
#define ZTR_TRawLaserPulse
#include "TObject.h"

class TRawLaserPulse : public TObject {

protected:

  Int_t   fNSample;              //Number of samples
  Int_t  *fSamples;              //[fNSample] sadc samples

  void Init();

public:

  TRawLaserPulse();
  TRawLaserPulse(Int_t,Int_t*);
  virtual ~TRawLaserPulse();
  void     Remove();

  Int_t    GetNSamples()              { return fNSample; }
  Int_t    GetSample(Int_t k) const   { return fSamples[k]; }
  //Int_t    GetSample(Int_t n) const { n%=fSamples; return fSamples[n]; }
  Int_t   *GetSamples();
  void     Print() const;
  void     SetLaserPulse(Int_t,Int_t*);
  ClassDef(TRawLaserPulse,1) //matacq data
};
#endif
