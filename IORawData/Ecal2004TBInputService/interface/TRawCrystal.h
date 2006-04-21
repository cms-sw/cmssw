#ifndef ZTR_TRawCrystal
#define ZTR_TRawCrystal
#include "TObject.h"
//#include "Config.h"

class TRawCrystal : public TObject {

private:

  Int_t  fHeader;   //FE channel header
  Int_t  fNSample;  //Number of samples
  Int_t *fSamples;  //[fNSample] values of the sampling ADC

  void Init();

public:

  static Int_t fgNSamplesCrystal;

  TRawCrystal();
  TRawCrystal(Int_t,Int_t*);
  virtual ~TRawCrystal();
  virtual void     Clear(const char *opt="");
  Int_t     Compare(const TObject *obj) const { 
              if (fHeader < ((TRawCrystal*)obj)->fHeader)
                return -1;
              else if (fHeader > ((TRawCrystal*)obj)->fHeader)
                return 1;
              else
                return 0; }
  Int_t    GetHeader()          { return fHeader;           }
  Int_t    GetNSamples()        { return fNSample; }
  Int_t    GetSample(Int_t n)   { return fSamples[n];       }
  Int_t   *GetSamples()         { return fSamples;          }
  Bool_t   IsSortable()  const  { return kTRUE;            }
  virtual void     Print(const char *opt=0)       const;
  void     SetCrystalRaw( Int_t, Int_t* );
  ClassDef(TRawCrystal,1)  //A cristal raw data from sampling adc
};
#endif
